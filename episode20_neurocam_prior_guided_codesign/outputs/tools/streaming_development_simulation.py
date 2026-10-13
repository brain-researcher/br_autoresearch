"""Saved-world conditional simulation with persistent streaming acquisition.

Caller-supplied physical parameters and a retained deterministic readout recipe
remain mandatory. A physical-action factory streams midpoint holds without
retaining the full schedule; the legacy supplied-action path retains its bounded
hold tuple. Conversion, digital dependency and output histories are streamed. The same held
input survives every storage seam. This implements no candidate comparison,
physical calibration or adaptive acquisition policy. Saved N1-N4 nuisances use
only the explicitly selected post-readout observation placement.
"""
from collections.abc import Mapping
from dataclasses import asdict
import math
import time

from cpe_array import NeurocamCPEArray
from circuit_noise_provider import SpectralCircuitNoise
from development_worlds import DevelopmentWorld
from development_simulation import _identity, _positive_int
from raw_input_hold import compile_midpoint_hold
from streaming_midpoint_hold import StreamingMidpointHoldPlan, compile_streaming_midpoint_hold
from raw_world_time import RawWorldTime
from observation_nuisance import SavedObservationNuisance
from sparse_array_rc import SolverControls
from surface_to_contact import registered_pad_geometry, build_contact_factors
from streaming_acquisition_readout import StreamingReadoutLimits
from streaming_array_acquisition import StreamingArrayAcquisition, StreamingArrayLimits
from streaming_input_plan import bind_hold_windows
from streaming_simulation_output import StreamingSimulationOutput


def run_streaming_development_simulation(*, world, geometry_id, short_axis, circuit_parameters,
        physical_actions, readout_window_factory, input_bindings, acquisition_window,
        final_publication_horizon, solver_controls, array_limits, readout_limits,
        epsilon_input_volts, max_hold_intervals, max_hold_work, max_elements,
        max_seconds, settings_reference, result_path, output_limits, noise_factory=None,
        execution_check=None, on_phase=None, preflight_only=False, on_preflight_progress=None):
    """Run the entire prescribed record once; persist terminal state or failure.

    There are no physical or numerical defaults. The input recipe behind
    ``settings_reference`` must retain all parameters, clocks, mappings,
    quantizers, finite limits and digital operations. Readout IDs follow the
    streaming API's sequential identities. Completion applies only to this
    specified engineering case, not the full EP20 panel or a calibrated device.

    Supply ``physical_actions`` as a replayable zero-argument factory for bounded
    full-record planning, or as the existing finite iterable for retained holds.
    Both use the same raw/physical boundaries and midpoint rule. Exact
    aperture/window splits then integrate that same held input without changing
    it. The analytic input-approximation certificate remains applicable to its
    subintervals; it is not a circuit-output error certificate. This avoids an
    eager collection of all conversion boundaries and makes the selected input
    approximation independent of the output buffer size.
    Optional noise_factory(array=..., check=...) binds one SpectralCircuitNoise
    before joint hold planning, without advancing the array. The frozen path,
    named physical ports, spectra and explicit A/V tolerances belong to the
    retained settings recipe. No physical noise law is inferred.
    Optional execution_check and on_phase callbacks expose the existing bounded
    operations to an enclosing allocation budget and cost reporter. They do not
    schedule work, change segmentation or advance physical state.
    Optional on_preflight_progress receives completed-window planning counters
    even if construction later fails; it does not replace final preflight status.

    Explicit preflight_only retains the complete declared hold/readout plan but
    stops after the acquisition constructor's exhaustive preflight. It creates
    no output writer and never calls runner.run(). Its preflight_complete status
    does not establish an electrical record or a scientific forward result.
    """
    if not isinstance(world, DevelopmentWorld):
        raise ValueError("an existing saved DevelopmentWorld is required")
    _identity(world.metadata, world.metadata.get("condition_id"), world.metadata.get("replicate_seed"))
    if not isinstance(settings_reference, str) or not settings_reference.strip():
        raise ValueError("retain a complete explicit settings reference")
    if noise_factory is not None and not callable(noise_factory):
        raise ValueError("noise_factory must be an explicit callable or None")
    if any(callback is not None and not callable(callback)
           for callback in (execution_check, on_phase, on_preflight_progress)):
        raise ValueError("execution, phase and preflight callbacks must be callable or None")
    if type(preflight_only) is not bool:
        raise ValueError("preflight_only must be an explicit boolean")
    base = dict(world_id=world.metadata["world_id"], independence_group=world.metadata["independence_group"],
                role=world.metadata["role"], settings_reference=settings_reference,
                actual_DAQ_configuration_qualified=False, calibrated_reference_model=False,
                candidate_scored=False, full_EP20_panel_complete=False,
                simulator_only_future_information=True, nuisance_applied=False)
    if (not isinstance(circuit_parameters, Mapping)
            or circuit_parameters.get("access_resistance_owner") != "assembler"
            or "daq_frontend" not in circuit_parameters):
        raise ValueError("explicit assembler-owned access and DAQ object or None required")
    if (not isinstance(solver_controls, SolverControls) or not isinstance(array_limits, StreamingArrayLimits)
            or not isinstance(readout_limits, StreamingReadoutLimits) or not callable(readout_window_factory)):
        raise ValueError("explicit streaming limits, solver controls and replayable readout recipe required")
    for name, value in (("max_hold_intervals", max_hold_intervals), ("max_hold_work", max_hold_work),
                        ("max_elements", max_elements)):
        _positive_int(value, name)
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("positive finite max_seconds required")
    required_output = {"max_payload_bytes", "max_records", "buffer_records", "identifier_bytes",
                       "max_dependencies_per_observation"}
    if not isinstance(output_limits, Mapping) or set(output_limits) != required_output:
        raise ValueError("all output limits must be supplied explicitly")
    started = time.monotonic()
    def check():
        if execution_check is not None:
            execution_check()
        if time.monotonic() - started >= max_seconds:
            raise RuntimeError("attempt elapsed cap reached between bounded operations")

    array = runner = writer = plan = geometry = nuisance = noise = None
    failure = output = readout_report = interrupted = None
    terminal, phase = "failed", "surface and hold preparation"
    def set_phase(value):
        nonlocal phase
        phase = value
        if on_phase is not None:
            on_phase(value)
    initial_state = initial_field = None
    try:
        set_phase(phase)
        check()
        geometry = registered_pad_geometry(geometry_id, short_axis=short_axis,
                                           max_elements=max_elements, check=check)
        contacts = build_contact_factors(world, geometry, max_elements=max_elements, check=check)
        raw_time = RawWorldTime(world, max_elements=max_elements, check=check)
        if world.metadata["condition_id"].startswith("N"):
            nuisance = SavedObservationNuisance(raw_time)
        elif world.nuisance_arrays:
            raise ValueError("Unexpected nuisance arrays cannot be silently ignored in a non-N world")
        base["observation_nuisance"] = None if nuisance is None else dict(nuisance.metadata)
        if noise_factory is not None:
            set_phase("electrical assembly and physical noise binding")
            check()
            array = NeurocamCPEArray(**dict(circuit_parameters))
            initial_state, initial_field = array.volts, array.previous_field_volts
            noise = noise_factory(array=array, check=check)
            if not isinstance(noise, SpectralCircuitNoise):
                raise ValueError("noise_factory must return a bound SpectralCircuitNoise")
            if array.volts != initial_state or array.previous_field_volts != initial_field:
                raise ValueError("noise binding must not advance the array")
            set_phase("joint field and physical noise hold preparation")
        base["physical_noise"] = None if noise is None else dict(noise.metadata)
        if callable(physical_actions):
            plan = compile_streaming_midpoint_hold(raw_time, contacts, physical_actions,
                support=acquisition_window, epsilon_input_volts=epsilon_input_volts,
                max_intervals=max_hold_intervals, max_work=max_hold_work, check=check,
                noise_hold_bound=None if noise is None else noise.hold_bound)
        else:
            plan = compile_midpoint_hold(raw_time, contacts, physical_actions,
                epsilon_input_volts=epsilon_input_volts, max_intervals=max_hold_intervals,
                max_work=max_hold_work, check=check,
                noise_hold_bound=None if noise is None else noise.hold_bound)
        factory = bind_hold_windows(plan, readout_window_factory, max_windows=array_limits.max_windows,
                                    max_window_fragments=array_limits.max_window_fragments, check=check)
        set_phase("electrical assembly")
        check()
        if array is None:
            array = NeurocamCPEArray(**dict(circuit_parameters))
            initial_state, initial_field = array.volts, array.previous_field_volts
        if not preflight_only:
            set_phase("exclusive output initialization")
            writer = StreamingSimulationOutput(result_path, settings_reference=settings_reference, **dict(output_limits))
            writer.begin(state_ids=array.state_ids, initial_state_volts=initial_state, initial_field_volts=initial_field,
                         metadata=dict(base, geometry=geometry.metadata, hold=dict(plan.report),
                                       assembly_resources=asdict(array.resources)))
        set_phase("whole-plan preflight")
        runner = StreamingArrayAcquisition(array=array, window_factory=factory,
            field_provider=plan.field_provider, input_bindings=input_bindings,
            acquisition_window=acquisition_window, final_publication_horizon=final_publication_horizon,
            controls=solver_controls, limits=array_limits, readout_limits=readout_limits,
            on_raw=(lambda _: None) if preflight_only else writer.emit_raw,
            on_observation=(lambda _: None) if preflight_only else writer.emit_observation,
            on_interval=(lambda _: None) if preflight_only else writer.emit_interval,
            check=check, observation_nuisance=nuisance,
            noise_provider=noise, on_preflight_progress=on_preflight_progress)
        if preflight_only:
            check()
            if array.volts != initial_state or array.previous_field_volts != initial_field:
                raise RuntimeError("preflight must preserve the original electrical state and field")
            set_phase("whole-plan preflight complete; electrical execution not selected")
            terminal = "preflight_complete"
        else:
            set_phase("persistent electrical/readout execution")
            readout_report = runner.run()
            check()
            terminal = "complete"
    except BaseException as error:
        if not isinstance(error, Exception):
            interrupted = error
        failure = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:1000],
                       executor_failure=None if runner is None else runner.failure)

    accounting = None if runner is None else runner.accounting_summary
    nuisance_count = (None if accounting is None else
                      accounting["readout_accounting"]["nuisance_observations_processed"])
    base["nuisance_applied"] = bool(nuisance_count)
    base["physical_noise"] = None if noise is None else dict(noise.metadata)
    hold_support = (None if plan is None else (plan.start, plan.end)
                    if isinstance(plan, StreamingMidpointHoldPlan)
                    else (plan.intervals[0].start, plan.intervals[-1].end))
    metadata = dict(base, status=terminal, failure=failure, readout=readout_report,
        accounting=accounting, preflight=None if runner is None else runner.preflight_summary,
        preflight_only=preflight_only, preflight_completed=runner is not None,
        full_record_completed=terminal == "complete",
        hold=None if plan is None else dict(plan.report),
        covers_full_16_second_record=hold_support == (0., 16.),
        confirmed_physical_time=None if runner is None else runner.physical_time,
        electrical_state_advanced=(True if accounting and accounting["completed_electrical_intervals"]
            else None if accounting and accounting["failed_call_operator_upper_bound"] else False),
        elapsed_seconds=time.monotonic() - started, elapsed_limit_seconds=max_seconds,
        elapsed_limit_scope=("checked_between_operations_and_solver_iterations_not_hard_interrupt"
            if array is not None and array.analog_conditioning is not None else
            "checked_between_operations_not_inside_solver_or_io"),
        dense_voltage_or_conversion_history_materialized=False,
        compact_full_hold_plan_retained=plan is not None and not isinstance(plan, StreamingMidpointHoldPlan),
        physical_action_factory_replayed=isinstance(plan, StreamingMidpointHoldPlan))
    if writer is not None:
        try:
            output = writer.finish(status=terminal, final_state_volts=array.volts,
                                   final_field_volts=array.previous_field_volts, metadata=metadata)
        except BaseException as error:
            if interrupted is None and not isinstance(error, Exception):
                interrupted = error
            # A failed writer cannot promote its prefix or be restarted. Keep
            # physical execution evidence separate from incomplete serialization.
            metadata["status"] = "failed"
            metadata["output_failure"] = dict(exception_type=type(error).__name__, message=str(error)[:1000])
            output = dict(path=str(writer.path), record_complete=False,
                          final_state_preserved_in_file=False)
        finally:
            writer.close()
    metadata["output"] = output
    if interrupted is not None:
        # Python-level interruption is propagated after best-effort failed
        # state serialization/closure. A hard kill can only leave an incomplete
        # container; it never grants restart or a completed-run claim.
        raise interrupted
    return metadata
