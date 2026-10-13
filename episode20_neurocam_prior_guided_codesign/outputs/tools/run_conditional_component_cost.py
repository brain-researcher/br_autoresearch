"""Bounded full-dimensional EP20 component timing, never a forward result.

Replay only saved B2/2001 and the original attempt002 noise realization. The
default component is the first actual electrical/aperture interval. The explicit
--first-two-actions mode continues the same array through the eight original
splits of its first two actions. Whole-record preflight, readout and scoring are
not performed. All circuit states and original input tolerances are retained.
The optional contact-block preconditioner changes only the iterative linear
solve; GMRES with stage-Jacobi remains the default. The optional triangular
contact-Schur solver preserves the original full-state acceptance gates and
reports its actual strategy and bounded direct-factor/solve metrics.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
from unittest.mock import patch


EPISODE_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = EPISODE_ROOT / "outputs/runs/development_worlds_job_47085635/panel/development_worlds.h5"
NOISE_PATH = EPISODE_ROOT / "outputs/runs/model_specification/conditional_forward_cost_attempt002/conditional_noise_draw.npz"
MAX_INCLUSIVE_SECONDS = 300.
WORK_SECONDS = 290.  # Reserve ten seconds for the bounded terminal report.
MAX_RSS_KIB = 13 * 1024**2 // 2
MAX_TOTAL_OPERATOR_PRODUCTS = 1000000000
WORK_FIELDS = (
    "field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
    "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
    "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
    "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule")


def _write_json(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > 1024**2:
        raise ValueError("component report exceeds its one MiB bound")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _retain_returned_diagnostics(target, result, np):
    kernel = result.kernel_interval
    endpoint = np.asarray(kernel.observation.end_volts)
    means = np.asarray(kernel.observation.mean_volts)
    target["returned_kernel"] = dict(integration_method=kernel.integration_method,
        radau_preconditioner=kernel.radau_preconditioner,
        radau_linear_solver=kernel.radau_linear_solver,
        radau_linear_solver_metadata=kernel.radau_linear_solver_metadata,
        state_count=len(endpoint), endpoint_min_volts=float(endpoint.min()),
        endpoint_max_volts=float(endpoint.max()), maximum_absolute_mean_volts=float(np.abs(means).max()),
        source_endpoint_volts=endpoint[4096:4160].tolist(), hold_endpoint_volts=endpoint[-24:].tolist(),
        source_mean_volts=means[4096:4160].tolist(), hold_mean_volts=means[-24:].tolist(),
        operator_products=kernel.operator_products, newton_iterations=kernel.newton_iterations,
        linear_iterations=kernel.linear_iterations, subintervals=kernel.subintervals,
        maximum_scaled_partition_error=kernel.maximum_scaled_partition_error,
        maximum_scaled_stage_residual=kernel.maximum_scaled_midpoint_residual,
        energy_balance_residual_joule=kernel.energy_balance_residual_joule,
        numerical_dissipation_joule=kernel.numerical_dissipation_joule,
        passive_norton_dissipation_joule=kernel.passive_norton_dissipation_joule,
        physical_noise_work_joule=list(kernel.physical_noise_work_joule),
        time_varying_current_work_joule=kernel.time_varying_current_work_joule,
        start_energy_joule=kernel.observation.start_energy_joule,
        end_energy_joule=kernel.observation.end_energy_joule,
        input_work_joule=kernel.observation.input_work_joule)
    target["physical_work"] = {name: getattr(result, name) for name in WORK_FIELDS}
    target["physical_energy"] = dict(before_boundary_joule=result.physical_energy_before_boundary_joule,
        after_boundary_joule=result.physical_energy_after_boundary_joule,
        interval_end_joule=result.physical_energy_end_joule)


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--first-two-actions", action="store_true",
        help="measure the eight contiguous original splits through5us on one persistent full array")
    parser.add_argument("--radau-preconditioner", choices=("stage_jacobi", "contact_block"),
        default="stage_jacobi", help="linear preconditioner only; all original equations and tolerances remain")
    parser.add_argument("--radau-linear-solver", choices=("gmres", "triangular_contact_schur"),
        default="gmres", help="explicit linear solver; original full-state residual/work gates and limits remain")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs") or not run_dir.is_dir()
            or any((run_dir / name).exists() for name in ("settings.json", "component_cost_report.json"))):
        raise RuntimeError("supply a fresh existing episode run directory; component outputs are exclusive")

    phase = "execution boundary"
    timings, diagnostics, counters = {}, {}, []
    interval_records, returned_results = [], []
    active_interval = None
    expected_interval_count = 8 if args.first_two_actions else 1
    allocation = array = noise = result = None
    before_state = before_field = None
    failure = None
    next_progress, next_rss = 0., 0.

    def check():
        nonlocal next_progress, next_rss
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= WORK_SECONDS:
            raise TimeoutError("290-second inclusive component-work deadline reached")
        if now >= next_rss:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
                raise MemoryError("component exceeds the unchanged6.5GiB peak-RSS bound")
            next_rss = now + .5
        if now >= next_progress:
            print(json.dumps(dict(event="component_progress", phase=phase, elapsed_seconds=elapsed)), flush=True)
            next_progress = now + 60.

    @contextmanager
    def measured(name):
        nonlocal phase
        previous, phase = phase, name
        beginning, completed = time.monotonic(), False
        try:
            check()
            yield
            check()
            completed = True
        finally:
            elapsed = time.monotonic() - beginning
            previous_timing = timings.get(name)
            timings[name] = dict(
                elapsed_seconds=elapsed + (0. if previous_timing is None else previous_timing["elapsed_seconds"]),
                completed=completed and (previous_timing is None or previous_timing["completed"]))
            if args.first_two_actions:
                timings[name]["calls"] = 1 if previous_timing is None else previous_timing["calls"] + 1
            if active_interval is not None:
                active_interval["timings"][name] = dict(elapsed_seconds=elapsed, completed=completed)
            if completed:
                phase = previous

    def interrupt(signum, frame):
        raise TimeoutError(f"bounded component interrupted by signal {signum}")

    previous_alarm = signal.signal(signal.SIGALRM, interrupt)
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, max(.001, WORK_SECONDS - (time.monotonic() - started)))
    try:
        # This module imports no numerical package before the actual shell,
        # Slurm allocation and thread environment have passed this boundary.
        from run_conditional_forward_cost import _compute_context
        allocation = _compute_context()
        if os.environ.get("BLIS_NUM_THREADS") != "1":
            raise RuntimeError("BLIS_NUM_THREADS must also be1 before numerical imports")
        with measured("scientific_imports"):
            import numpy as np
            import conditioned_array_solver as solver_module
            import cpe_array as array_module
            from conditional_acquisition_recipe import (acquisition_recipe_metadata,
                build_conditional_frontend, build_conditional_schedule, conditional_noise_cell)
            from conditional_forward_recipe import (build_conditional_circuit_parameters,
                build_conditional_noise, expected_state_ids, metadata)
            from conditioned_array_solver import ConditionedSolverControls
            from development_simulation import load_saved_development_world
            from raw_input_hold import compile_midpoint_hold
            from raw_world_time import RawWorldTime
            from surface_to_contact import registered_pad_geometry, build_contact_factors

        solver = ConditionedSolverControls(relative_tolerance=1e-5, voltage_atol=1e-8,
            charge_atol_coulomb=1e-20, max_refinements=8, max_operator_products=200000,
            linear_relative_tolerance=1e-9, linear_charge_atol_coulomb=1e-22,
            energy_atol_joule=1e-18, max_newton_iterations=12,
            max_linear_iterations=10000, gmres_restart=40, max_workspace_bytes=512 * 1024**2,
            integration_method="radau_iia_2", radau_preconditioner=args.radau_preconditioner,
            radau_linear_solver=args.radau_linear_solver)
        settings = dict(scope=("full_dimensional_first_two_actions_engineering_cost_only" if args.first_two_actions
                else "full_dimensional_first_interval_engineering_cost_only"),
            selected_interval=("first_two_original_actions_eight_contiguous_electrical_splits" if args.first_two_actions
                else "first_field_hold_split_at_first_actual_aperture_boundary"),
            first_two_actions=args.first_two_actions, expected_electrical_intervals=expected_interval_count,
            selected_radau_preconditioner=args.radau_preconditioner,
            selected_radau_linear_solver=args.radau_linear_solver,
            maximum_total_operator_products=MAX_TOTAL_OPERATOR_PRODUCTS,
            allocation=allocation, electrical=metadata(), acquisition=dict(acquisition_recipe_metadata()),
            world_path=str(WORLD_PATH), world_selection=dict(condition_id="B2", replicate_seed=2001),
            reused_noise_path=str(NOISE_PATH), original_noise_realization_reused=True,
            solver_controls=asdict(solver), field_hold_absolute_tolerance_volts=1e-8,
            noise_current_hold_absolute_tolerance_amp=1e-12, noise_voltage_hold_absolute_tolerance_volts=1e-8,
            max_spectral_evaluations=200000000, max_spectral_term_evaluations=200000000000000,
            max_hold_intervals=40000000, max_hold_work=100000000,
            max_elements=2000000, world_numeric_bytes_limit=64 * 1024**2, world_metadata_bytes_limit=1024**2,
            inclusive_limit_seconds=MAX_INCLUSIVE_SECONDS, work_deadline_seconds=WORK_SECONDS,
            maximum_peak_rss_kib=MAX_RSS_KIB, full_record_preflight_performed=False,
            full_record_completed=False, raw_conversion_records_produced=0,
            calibrated_reference=False, candidate_scored=False,
            component_array_commit="actual_CPE_commit_after_its_network_preview_and_physical_accounting",
            timing_instrumentation="scoped_callbacks_only_no_numerical_operator_replacement",
            timing_scope="nested_wall_measurements_not_additive",
            zero_initial_state_is_not_stationary_equilibrium=True)
        with measured("retaining_settings"):
            _write_json(run_dir / "settings.json", settings)
        with measured("saved_world_loading"):
            world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
                max_elements=2000000, max_array_bytes=64 * 1024**2,
                max_metadata_bytes=1024**2, check=check)
            if world.nuisance_arrays:
                raise ValueError("selected B2 component cannot silently omit nuisance arrays")
        with measured("full_geometry_and_raw_field_binding"):
            geometry = registered_pad_geometry("G0", short_axis="x", max_elements=2000000, check=check)
            contacts = build_contact_factors(world, geometry, max_elements=2000000, check=check)
            raw_time = RawWorldTime(world, max_elements=2000000, check=check)
        with measured("full_circuit_and_schedule_construction"):
            frontend = build_conditional_frontend()
            parameters = build_conditional_circuit_parameters(daq_frontend=frontend, check=check)
            array = array_module.NeurocamCPEArray(**parameters)
            if array.state_ids != expected_state_ids() or len(array.state_ids) != 69848:
                raise ValueError("component must retain all69848 original circuit states")
            before_state, before_field = array.volts, array.previous_field_volts
            schedule = build_conditional_schedule(frontend, state_ids=array.state_ids, check=check)
            action_iterator = schedule.physical_actions_for_cycle(0)
            actions = tuple(next(action_iterator) for _ in range(2 if args.first_two_actions else 1))
            action = actions[0]
            readout_window = schedule.readout_window(0)
            diagnostics["assembly_resources"] = asdict(array.resources)
            diagnostics["physical_pads"] = 4096
            diagnostics["source_lines"] = 64
            diagnostics["physical_ADC_groups"] = len(frontend.groups)
        with measured("original_noise_replay_and_binding"):
            noise = build_conditional_noise(array=array, check=check, retained_draw_path=None,
                current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
                max_evaluations=200000000, max_term_evaluations=200000000000000,
                reuse_draw_path=NOISE_PATH, cubic_cell_for_key=conditional_noise_cell)
            diagnostics["spectral_before"] = noise.spectrum.accounting()
            diagnostics["port_count_before"] = noise.layout.numeric_value_count
        with measured("first_original_action_hold_selection"):
            # This small component plan selects the same first held field as
            # the full streaming plan: aperture boundaries split it later,
            # without recentering its field key or original noise polynomial.
            plan = compile_midpoint_hold(raw_time, contacts, actions, epsilon_input_volts=1e-8,
                max_intervals=40000000, max_work=100000000, check=check,
                noise_hold_bound=noise.hold_bound)
            first = plan.intervals[0]
            if (action.start != 0. or action.end != 2.5e-6 or len(plan.intervals) != 2 * len(actions)
                    or first.start != 0. or first.end != 1.25e-6 or first.field_key != .625e-6):
                raise ValueError("first action/hold differs from the measured attempt003 decomposition")
            endpoints = [first.end]
            endpoints.extend(point for conversion in readout_window.conversions
                for point in (conversion.aperture_start, conversion.aperture_end)
                if first.start < point < first.end)
            end = min(endpoints)
            if end != .5e-6:
                raise ValueError("first actual aperture split differs from the selected original recipe")
            diagnostics["component_hold_plan"] = dict(plan.report)
            diagnostics["component_interval"] = dict(start_seconds=first.start, end_seconds=end,
                duration_seconds=end - first.start, field_key_seconds=first.field_key,
                gate_row=first.gate_row, observed_sources=[0],
                initial_boundary_charge_entries=len(first.injected_charge_coulomb))
            splits = [(0, 0, first.start, end)]
            if args.first_two_actions:
                if (actions[1].start != action.end or actions[1].end != 5e-6
                        or any(item.gate_row != 0 for item in actions)
                        or tuple(len(item.injected_charge_coulomb) for item in actions) != (88, 24)
                        or any(binding.kind != "hold" for binding in schedule.input_bindings)):
                    raise ValueError("first two actions differ from the selected row0/physical-hold/charge recipe")
                # The holds are compiled before aperture splitting. Preserve
                # their exact represented midpoint arithmetic and original keys.
                expected_holds = []
                for item in actions:
                    middle = item.start + (item.end - item.start) / 2
                    expected_holds.extend(((item.start, middle, item.start + (middle - item.start) / 2),
                        (middle, item.end, middle + (item.end - middle) / 2)))
                if tuple((item.start, item.end, item.field_key) for item in plan.intervals) != tuple(expected_holds):
                    raise ValueError("first two actions no longer have the four selected original field holds")
                splits = []
                for hold_index, held in enumerate(plan.intervals):
                    points = {held.start, held.end}
                    points.update(point for conversion in readout_window.conversions
                        for point in (conversion.aperture_start, conversion.aperture_end)
                        if held.start < point < held.end)
                    points = sorted(points)
                    splits.extend((hold_index // 2, hold_index, left, right)
                        for left, right in zip(points, points[1:]))
                expected_edges = (0., .5e-6, expected_holds[0][1], 2e-6, 2.5e-6,
                    3e-6, expected_holds[2][1], 4.5e-6, 5e-6)
                if (len(splits) != 8 or tuple(item[2] for item in splits) + (splits[-1][3],) != expected_edges):
                    raise ValueError("first two actions no longer have the eight selected original aperture/hold splits")
                diagnostics["component_intervals"] = interval_records
                diagnostics["selected_original_edges_seconds"] = expected_edges
                diagnostics["selected_original_field_keys_seconds"] = [item.field_key for item in plan.intervals]
                diagnostics["boundary_charge_entries_per_action"] = [88, 24]
            diagnostics["setup_elapsed_seconds_before_component_intervals"] = time.monotonic() - started
        original_coefficients = noise.spectrum.cubic_coefficients
        def timed_coefficients(origin):
            with measured("spectral_cubic_coefficient_kernel"):
                return original_coefficients(origin)
        original_forcing = array_module.RadauPolynomialCircuitForcing
        original_counter = solver_module._Counter
        original_advance = array._network.advance

        def timed_forcing(*values, **named):
            with measured("original_port_coefficient_projection"):
                return original_forcing(*values, **named)

        def owned_counter(*values, **named):
            if len(counters) != len(returned_results):
                raise RuntimeError("each component interval admits one original network-counter instance only")
            counter = original_counter(*values, **named)
            active_interval["solver_counter_index"] = len(counters)
            counters.append(counter)
            return counter

        def timed_network_advance(**named):
            if named.get("commit") is not False:
                raise RuntimeError("the actual CPE network stage must remain its existing noncommitting preview")
            diagnostics["network_preview_commit_argument"] = False
            active_interval["network_preview_commit_argument"] = False
            with measured("actual_radau_network_preview"):
                return original_advance(**named)

        current_action = current_hold = None
        with patch.object(array_module, "RadauPolynomialCircuitForcing", new=timed_forcing), \
             patch.object(solver_module, "_Counter", new=owned_counter), \
             patch.object(array._network, "advance", new=timed_network_advance):
            for action_index, hold_index, start, end in splits:
                phase = "component interval setup"
                check()
                held = plan.intervals[hold_index]
                active_interval = dict(index=len(interval_records), action_index=action_index,
                    hold_index=hold_index, start_seconds=start, end_seconds=end,
                    duration_seconds=end - start, field_key_seconds=held.field_key,
                    gate_row=held.gate_row, observed_sources=[0], timings={},
                    array_call_returned=False, component_commit_confirmed=False,
                    port_count_before=noise.layout.numeric_value_count)
                interval_records.append(active_interval)
                interval_started = time.monotonic()
                try:
                    if action_index != current_action:
                        with measured("original_cell_cubic_materialization"):
                            with patch.object(noise.spectrum, "cubic_coefficients", new=timed_coefficients):
                                polynomial = noise(held.field_key)
                        with measured("original_polynomial_interval_validation"):
                            array.validate_noise_interval(polynomial, daq_selected_sources=held.daq_selected_sources)
                            if (polynomial.support_start_seconds != actions[action_index].start
                                    or polynomial.support_end_seconds != actions[action_index].end):
                                raise ValueError("component polynomial must retain its complete original physical cell")
                        current_action = action_index
                    if hold_index != current_hold:
                        with measured("original_field_hold_materialization"):
                            field = plan.field_provider(held.field_key)
                        current_hold = hold_index
                    active_interval["polynomial"] = dict(origin_seconds=polynomial.origin_seconds,
                        support_start_seconds=polynomial.support_start_seconds,
                        support_end_seconds=polynomial.support_end_seconds,
                        numeric_bytes=polynomial.sample_numeric_bytes,
                        normalized_original_cell_remainder_bound=noise.hold_bound(held.start, held.end, held.field_key))
                    charge_entries = held.injected_charge_coulomb if start == held.start else ()
                    charge = np.zeros(len(array.state_ids))
                    for node, value in charge_entries:
                        charge[node] = value
                    active_interval["boundary_charge_entries"] = len(charge_entries)
                    remaining = MAX_TOTAL_OPERATOR_PRODUCTS - sum(counter.products for counter in counters)
                    if remaining < 1:
                        raise ValueError("cumulative component propagation-operator cap exhausted")
                    interval_controls = replace(solver,
                        max_operator_products=min(solver.max_operator_products, remaining))
                    active_interval["operator_product_limit"] = interval_controls.max_operator_products
                    with measured("actual_full_array_component_advance"):
                        result = array.advance_row(gate_row=held.gate_row, duration_seconds=end - start,
                            field_volts=field, voltage_reference_id=held.voltage_reference_id,
                            observed_sources=(0,), injected_charge_coulomb=charge, controls=interval_controls,
                            daq_selected_sources=held.daq_selected_sources, check=check,
                            noise_sample=polynomial, interval_start_seconds=start)
                        # Retain the successful return before the measurement's
                        # post-call deadline check can interrupt accounting.
                        returned_results.append((active_interval, result))
                    with measured("retaining_returned_electrical_diagnostics"):
                        _retain_returned_diagnostics(active_interval, result, np)
                finally:
                    active_interval["elapsed_seconds_including_interval_setup"] = time.monotonic() - interval_started
                    active_interval["port_count_after"] = noise.layout.numeric_value_count
                active_interval = None
        phase = "component measurement complete"
    except BaseException as error:
        failure = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:1000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)

    if noise is not None:
        diagnostics["spectral_after"] = noise.spectrum.accounting()
        diagnostics["port_count_after"] = noise.layout.numeric_value_count
    if array is not None and before_state is not None:
        diagnostics["component_state_changed"] = array.volts != before_state
        diagnostics["component_field_changed"] = array.previous_field_volts != before_field
    # At most eight compact summaries, including a return immediately preceding
    # the deadline. No new solve, waveform or observation is evaluated here.
    for item, returned in returned_results:
        item["array_call_returned"] = item["component_commit_confirmed"] = True
        if "physical_energy" not in item:
            _retain_returned_diagnostics(item, returned, np)
    for item in interval_records:
        counter_index = item.get("solver_counter_index")
        counter = None if counter_index is None else counters[counter_index]
        item["actual_solver_counter"] = dict(counter_created=counter is not None,
            operator_products=0 if counter is None else counter.products,
            newton_iterations=0 if counter is None else counter.newton,
            linear_iterations=0 if counter is None else counter.linear)
    if not args.first_two_actions and interval_records:
        for key in ("polynomial", "returned_kernel", "physical_work", "physical_energy"):
            if key in interval_records[0]:
                diagnostics[key] = interval_records[0][key]
    diagnostics["actual_solver_counter"] = dict(counter_created=bool(counters),
        counter_instances=len(counters),
        operator_products=sum(counter.products for counter in counters),
        newton_iterations=sum(counter.newton for counter in counters),
        linear_iterations=sum(counter.linear for counter in counters),
        maximum_total_operator_products=MAX_TOTAL_OPERATOR_PRODUCTS,
        scope="actual_existing_counter_including_unreturned_solve_not_factorization_or_C_products")
    confirmed_count = len(returned_results)
    confirmed_frontier = 0. if not returned_results else returned_results[-1][0]["end_seconds"]
    elapsed = time.monotonic() - started
    complete = failure is None and confirmed_count == expected_interval_count
    report = dict(status="component_complete" if complete else "component_failed",
        phase=phase, failure=failure, allocation=allocation, elapsed_seconds_before_report=elapsed,
        inclusive_limit_seconds=MAX_INCLUSIVE_SECONDS, work_deadline_seconds=WORK_SECONDS,
        work_and_diagnostics_within_inclusive_limit=elapsed <= MAX_INCLUSIVE_SECONDS,
        peak_rss_kib=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), timings=timings,
        timing_scope="nested_wall_measurements_not_additive",
        diagnostics=diagnostics, component_interval_returned=bool(confirmed_count),
        component_commit_confirmed=bool(confirmed_count), first_two_actions=args.first_two_actions,
        selected_radau_preconditioner=args.radau_preconditioner,
        selected_radau_linear_solver=args.radau_linear_solver,
        confirmed_radau_linear_solvers=sorted({
            value.kernel_interval.radau_linear_solver for _, value in returned_results}),
        expected_electrical_intervals=expected_interval_count,
        confirmed_returned_intervals=confirmed_count, attempted_intervals=len(interval_records),
        confirmed_physical_frontier_seconds=confirmed_frontier,
        confirmed_operator_products=sum(value.kernel_interval.operator_products for _, value in returned_results),
        confirmed_frontier_scope="successfully_returned_CPE_intervals_only_no_claim_for_an_unreturned_call",
        full_record_preflight_performed=False, full_record_completed=False,
        raw_conversion_records_produced=0, observations_produced=0, calibrated_reference=False,
        candidate_scored=False, payload_scope="saved_B2_seed2001_and_original_attempt002_noise_only",
        scope="engineering_component_cost_not_shortened_record_scientific_forward_result")
    _write_json(run_dir / "component_cost_report.json", report)
    print(json.dumps(dict(event="component_terminal", status=report["status"], phase=phase,
        elapsed_seconds=time.monotonic() - started,
        selected_radau_linear_solver=args.radau_linear_solver,
        confirmed_radau_linear_solvers=report["confirmed_radau_linear_solvers"],
        report=str(run_dir / "component_cost_report.json"))), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
