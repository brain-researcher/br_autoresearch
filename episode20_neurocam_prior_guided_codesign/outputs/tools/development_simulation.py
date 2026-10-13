"""One bounded, caller-parameterized EP20 surface/array/readout attempt.

This module consumes a retained development world; it never generates one,
selects physical parameters, fits anchors, scores a design or drives a policy.
The caller must retain the exact circuit/timing/readout inputs under the supplied
settings_reference. A completed result means this finite passive mechanism ran,
not that NeuroCam, a reference model or the full EP20 comparison is qualified.
N1--N4 remain unresolved until their observation-side placement is selected.

Execute on coordinated compute only. Array/count/operator limits are explicit;
wall time is checked between bounded operations, not a hard interruption of a
sparse solve. Slurm still supplies the process memory and hard wall-time limits.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
import json
import math
from pathlib import Path
import time

import numpy as np

from acquisition_readout import RawConversion
from array_acquisition import AcquisitionLimits, ArrayAcquisition
from cpe_array import NeurocamCPEArray
from development_worlds import CONDITIONS, RECIPE, REPLICATE_SEEDS, DevelopmentWorld
from raw_input_hold import compile_midpoint_hold
from circuit_noise_provider import SpectralCircuitNoise
from raw_world_time import RawWorldTime
from sparse_array_rc import SolverControls
from surface_to_contact import registered_pad_geometry, build_contact_factors


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def _identity(metadata, condition, seed):
    if (condition not in CONDITIONS or seed not in REPLICATE_SEEDS
            or metadata.get("recipe") != RECIPE
            or metadata.get("world_id") != f"{condition}:{seed}"
            or metadata.get("condition_id") != condition
            or metadata.get("replicate_seed") != seed
            or metadata.get("role") != "development_preparation_unassigned"
            or metadata.get("independence_group") != f"development_replicate_{seed}"
            or metadata.get("final_world_access") is not False):
        raise ValueError("only the exact selected unassigned development world is admitted")


def load_saved_development_world(path, *, condition_id, replicate_seed, max_elements,
                                 max_array_bytes, max_metadata_bytes, check=None):
    """Read one exact completed group from the generator's existing HDF5 file.

    Numeric shapes, dtypes and aggregate bytes are checked before materializing
    arrays. The trusted generator's scalar variable-length metadata string has
    its byte cap checked immediately after reading; HDF5 does not expose its
    decoded size in the high-level dataset shape. No other world is opened.
    External/soft links, virtual datasets and external numeric storage are not
    followed. These constraints prevent this loader from crossing world roles.
    """
    import h5py
    for name, value in (("max_elements", max_elements), ("max_array_bytes", max_array_bytes),
                        ("max_metadata_bytes", max_metadata_bytes)):
        _positive_int(value, name)
    if condition_id not in CONDITIONS or type(replicate_seed) is not int or replicate_seed not in REPLICATE_SEEDS:
        raise ValueError("explicit registered condition and retained development seed required")
    check = (lambda: None) if check is None else check
    if not callable(check):
        raise ValueError("check must be callable")
    key = f"{condition_id}_seed_{replicate_seed}"
    def local(parent, name, cls):
        if not isinstance(parent.get(name, getlink=True), h5py.HardLink):
            raise ValueError("saved-world entries must be local hard links")
        value = parent[name]
        if not isinstance(value, cls):
            raise ValueError("saved-world entry has an unexpected type")
        if isinstance(value, h5py.Dataset) and (value.is_virtual or value.external):
            raise ValueError("virtual/external data are outside this saved-world loader")
        return value
    check()
    with h5py.File(path, "r") as container:
        group = local(container, key, h5py.Group)
        if not bool(group.attrs.get("complete", False)):
            raise ValueError("selected saved world is not complete")
        meta = local(group, "metadata_json", h5py.Dataset)
        if meta.shape != ():
            raise ValueError("one scalar generator metadata record is required")
        text = meta.asstr()[()]
        if len(text.encode("utf-8")) > max_metadata_bytes:
            raise ValueError("selected-world metadata exceeds its finite byte cap")
        metadata = json.loads(text)
        _identity(metadata, condition_id, replicate_seed)
        if metadata.get("production_grid") is not True or metadata.get("grid_shape_time_y_x") != [8000, 384, 384]:
            raise ValueError("the retained production grid is required")
        modes = metadata.get("factor_modes")
        if type(modes) is not int or modes < 1:
            raise ValueError("explicit positive factor rank required")
        datasets = {}
        for name, shape in (("spatial_modes", (modes, 384, 384)),
                            ("raw_drivers", (modes, 8000)),
                            ("primary_band_drivers", (modes, 8000))):
            value = local(group, name, h5py.Dataset)
            if value.shape != shape or value.dtype != np.dtype("float64"):
                raise ValueError("stored factor dimensions/dtype disagree with the selected recipe")
            datasets[name] = value
        nuisance_group = local(group, "observation_side_nuisance_inputs", h5py.Group)
        expected = ({"spatially_common_line_uv": ((8000,), np.dtype("float64"))}
                    if condition_id in ("N1", "N2") else
                    {"missing_sites_packed_little": ((8000, 512), np.dtype("uint8"))}
                    if condition_id == "N3" else {})
        if set(nuisance_group) != set(expected):
            raise ValueError("stored nuisance channels disagree with the selected condition")
        for name, (shape, dtype) in expected.items():
            value = local(nuisance_group, name, h5py.Dataset)
            if value.shape != shape or value.dtype != dtype:
                raise ValueError("stored nuisance shape/dtype mismatch")
            datasets[name] = value
        if (any(value.size > max_elements for value in datasets.values())
                or sum(value.size * value.dtype.itemsize for value in datasets.values()) > max_array_bytes):
            raise ValueError("saved world exceeds explicit numeric loading limits")
        loaded = {}
        for name, value in datasets.items():
            check()
            array = value[...]
            if not np.isfinite(array).all():
                raise ValueError("nonfinite retained world array")
            array.flags.writeable = False
            loaded[name] = array
        check()
    return DevelopmentWorld(loaded.pop("spatial_modes"), loaded.pop("raw_drivers"),
                            loaded.pop("primary_band_drivers"), metadata, loaded)


class _ExecutionLimit(RuntimeError):
    pass


class _Sum:
    """Constant-space compensated sum of returned component diagnostics."""
    def __init__(self):
        self.total = self.correction = 0.

    def add(self, value):
        value = float(value)
        total = self.total + value
        self.correction += ((self.total - total) + value if abs(self.total) >= abs(value)
                            else (value - total) + self.total)
        self.total = total
        if not math.isfinite(self.value):
            raise ValueError("nonrepresentable aggregate component diagnostic")

    @property
    def value(self):
        return self.total + self.correction


class _AccountingArray(NeurocamCPEArray):
    """Same full physics; streaming diagnostics and a remaining-work cap only."""
    def __init__(self, *, total_operator_limit, checkpoint, **parameters):
        self.checkpoint, self.operator_limit = checkpoint, total_operator_limit
        self.operator_products = self.completed_calls = self.failed_call_operator_upper_bound = 0
        self.first_energy = self.last_energy = None
        self.maximum_local_error = self.solver_seconds = self.maximum_step_seconds = 0.
        self.failed_solver_seconds = 0.
        self.accounting_complete = True
        self._numerical_integration = None
        self.sums = {key: _Sum() for key in (
            "field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
            "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
            "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
            "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule",
            "physical_boundary_energy_change_joule",
            "physical_interval_energy_change_joule")}
        super().__init__(**parameters)

    def advance_row(self, **kwargs):
        self.checkpoint()
        remaining = self.operator_limit - self.operator_products
        if remaining < 1:
            raise _ExecutionLimit("total propagation-operator budget exhausted before next interval")
        original = kwargs["controls"]
        limited = replace(original, max_operator_products=min(original.max_operator_products, remaining))
        kwargs["controls"] = limited
        started = time.monotonic()
        try:
            result = super().advance_row(**kwargs)
        except BaseException:
            elapsed = time.monotonic() - started
            self.failed_solver_seconds += elapsed
            self.solver_seconds += elapsed
            # The component did not return a count; do not misreport failed
            # propagation work as zero. No replay is performed by this runner.
            self.failed_call_operator_upper_bound = limited.max_operator_products
            raise
        elapsed = time.monotonic() - started
        self.solver_seconds += elapsed
        self.maximum_step_seconds = max(self.maximum_step_seconds, elapsed)
        self.completed_calls += 1
        self.operator_products += result.kernel_interval.operator_products
        self.maximum_local_error = max(self.maximum_local_error,
                                      result.kernel_interval.maximum_scaled_partition_error)
        if self.first_energy is None:
            self.first_energy = result.physical_energy_before_boundary_joule
        self.last_energy = result.physical_energy_end_joule
        try:
            if getattr(result.kernel_interval, "integration_method", None) == "radau_iia_2":
                # Same accepted-interval diagnostic as the streaming runner;
                # numerical loss is never a physical heat/source-work entry.
                if self._numerical_integration is None:
                    self._numerical_integration = dict(integration_method="radau_iia_2",
                        returned_electrical_intervals=0, numerical_dissipation_joule=0.,
                        complete_for_returned_intervals=True,
                        scope="accepted_intervals_only_not_physical_heat_or_source_work")
                try:
                    value = float(result.kernel_interval.numerical_dissipation_joule)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("returned numerical dissipation must be finite and nonnegative")
                    total = math.fsum((self._numerical_integration["numerical_dissipation_joule"], value))
                    if not math.isfinite(total) or total < 0:
                        raise ValueError("nonrepresentable accumulated numerical dissipation")
                    self._numerical_integration["numerical_dissipation_joule"] = total
                    self._numerical_integration["returned_electrical_intervals"] += 1
                except BaseException:
                    self._numerical_integration["complete_for_returned_intervals"] = False
                    raise
            for key in ("field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
                        "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
                        "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
                        "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule"):
                self.sums[key].add(getattr(result, key))
            self.sums["physical_boundary_energy_change_joule"].add(
                result.physical_energy_after_boundary_joule - result.physical_energy_before_boundary_joule)
            self.sums["physical_interval_energy_change_joule"].add(
                result.physical_energy_end_joule - result.physical_energy_after_boundary_joule)
        except BaseException:
            self.accounting_complete = False  # Electrical state already committed.
            raise
        return result

    def accounting(self):
        report = dict(**{key: value.value for key, value in self.sums.items()},
                    physical_initial_energy_joule=self.first_energy,
                    physical_final_energy_joule=self.last_energy,
                    returned_electrical_intervals=self.completed_calls,
                    confirmed_operator_products=self.operator_products,
                    failed_call_operator_products=None if self.failed_call_operator_upper_bound else 0,
                    failed_call_operator_upper_bound=self.failed_call_operator_upper_bound,
                    total_operator_limit=self.operator_limit,
                    maximum_local_scaled_partition_error=self.maximum_local_error,
                    aggregate_accounting_complete_for_returned_calls=self.accounting_complete,
                    measured_solver_seconds=self.solver_seconds,
                    measured_failed_solver_seconds=self.failed_solver_seconds,
                    maximum_returned_step_seconds=self.maximum_step_seconds,
                    modeled_complete_device_power_watt=None,
                    scope=("declared_passive_circuit_storage_and_port_work_only"
                           if self.analog_conditioning is None else
                           "declared_conditioned_circuit_storage_and_port_work_not_supply_consumption"))
        if self._numerical_integration is not None:
            report["numerical_integration"] = dict(self._numerical_integration)
        return report


def run_development_simulation(*, world, geometry_id, short_axis, circuit_parameters,
                               physical_actions, input_bindings, conversions, returned_samples,
                               observations, acquisition_window, final_publication_horizon,
                               solver_controls, acquisition_limits, epsilon_input_volts,
                               max_hold_work, max_elements, max_total_operator_products,
                               max_seconds, settings_reference, noise_factory=None):
    """Execute one prescribed conditional mechanism, preserving all array states.

    All exact physical parameters, initial states and readout choices are caller
    inputs. ``settings_reference`` must name their retained recipe; the output
    does not duplicate or infer that recipe. Supply raw conversions, not rates
    from which this function would have to guess apertures or channel mapping.
    Optional noise_factory(array=..., check=...) binds one SpectralCircuitNoise
    to the assembled topology before hold planning. It must preserve array
    state and retain its explicit spectra, draws, ports and per-port tolerances
    in the settings recipe; it cannot infer missing physical noise as zero.

    Returns a terminal result on completion or execution/preparation failure;
    it never returns a mutable runner or resumable electrical object. Incomplete
    attempts expose final state and confirmed accounting, not unfinished raw
    readout codes. Invalid arguments fail before execution. No auto-retry.
    """
    if not isinstance(world, DevelopmentWorld):
        raise ValueError("an existing DevelopmentWorld is required; no regeneration is performed")
    _identity(world.metadata, world.metadata.get("condition_id"), world.metadata.get("replicate_seed"))
    if not isinstance(settings_reference, str) or not settings_reference.strip():
        raise ValueError("retain the complete caller inputs under an explicit settings_reference")
    base = dict(world_id=world.metadata["world_id"], independence_group=world.metadata["independence_group"],
                role=world.metadata["role"], settings_reference=settings_reference,
                actual_DAQ_configuration_qualified=False, calibrated_reference_model=False,
                candidate_scored=False, full_EP20_panel_complete=False,
                simulator_only_future_information=True, nuisance_applied=False)
    if world.metadata["condition_id"].startswith("N") or world.nuisance_arrays:
        return dict(base, status="unresolved", reason="observation-side nuisance placement is not implemented",
                    electrical_state_advanced=False, readout=None)
    if (not isinstance(circuit_parameters, Mapping)
            or circuit_parameters.get("access_resistance_owner") != "assembler"):
        raise ValueError("surface inputs require explicitly assembler-owned access resistance")
    if "daq_frontend" not in circuit_parameters:
        raise ValueError("explicit daq_frontend object or None is required; no DAQ absence is inferred")
    if noise_factory is not None and not callable(noise_factory):
        raise ValueError("noise_factory must be an explicit callable or None")
    if not isinstance(solver_controls, SolverControls) or not isinstance(acquisition_limits, AcquisitionLimits):
        raise ValueError("explicit numerical controls and acquisition limits required")
    for name, value in (("max_hold_work", max_hold_work), ("max_elements", max_elements),
                        ("max_total_operator_products", max_total_operator_products)):
        _positive_int(value, name)
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("positive finite max_seconds required")
    started = time.monotonic()
    def checkpoint():
        if time.monotonic() - started >= max_seconds:
            raise _ExecutionLimit("attempt elapsed-time cap reached; no next operation is started")
    raw = []
    for conversion in conversions:
        checkpoint()
        if not isinstance(conversion, RawConversion) or len(raw) >= acquisition_limits.max_conversions:
            raise ValueError("explicit raw conversion plan exceeds its type/count limits")
        raw.append(conversion)
    # The hold planner limits boundary entries as well as intervals. Deduplicate
    # here because multiple simultaneous ADCs can have identical apertures.
    aperture_edges = sorted({edge for c in raw for edge in (c.aperture_start, c.aperture_end)})
    array = runner = plan = geometry = noise = None
    initial_state = initial_field = None
    report = None
    phase, failure, terminal = "surface preparation", None, "failed"
    try:
        geometry = registered_pad_geometry(geometry_id, short_axis=short_axis,
                                           max_elements=max_elements, check=checkpoint)
        contacts = build_contact_factors(world, geometry, max_elements=max_elements, check=checkpoint)
        raw_time = RawWorldTime(world, max_elements=max_elements, check=checkpoint)
        if noise_factory is not None:
            phase = "electrical assembly and physical noise binding"
            checkpoint()
            array = _AccountingArray(total_operator_limit=max_total_operator_products,
                                     checkpoint=checkpoint, **dict(circuit_parameters))
            initial_state, initial_field = array.volts, array.previous_field_volts
            noise = noise_factory(array=array, check=checkpoint)
            if not isinstance(noise, SpectralCircuitNoise):
                raise ValueError("noise_factory must return a bound SpectralCircuitNoise")
            if array.volts != initial_state or array.previous_field_volts != initial_field:
                raise ValueError("noise binding must not advance the array")
            phase = "joint field and physical noise hold preparation"
        plan = compile_midpoint_hold(raw_time, contacts, physical_actions,
            epsilon_input_volts=epsilon_input_volts, max_intervals=acquisition_limits.max_physical_intervals,
            max_work=max_hold_work, additional_boundaries_s=aperture_edges, check=checkpoint,
            noise_hold_bound=None if noise is None else noise.hold_bound)
        # Since every aperture is already a hold boundary, the executor must
        # not add further splits. Reject before constructing the full circuit.
        if len(plan.intervals) > acquisition_limits.max_split_intervals:
            raise ValueError("compiled intervals exceed the explicit split limit")
        phase = "electrical assembly"
        checkpoint()
        if array is None:
            array = _AccountingArray(total_operator_limit=max_total_operator_products,
                                     checkpoint=checkpoint, **dict(circuit_parameters))
            initial_state, initial_field = array.volts, array.previous_field_volts
        phase = "complete schedule and field validation"
        def provider(key):
            checkpoint()
            return plan.field_provider(key)
        runner = ArrayAcquisition(array=array, physical_intervals=plan.intervals,
            input_bindings=input_bindings, conversions=raw, returned_samples=returned_samples,
            observations=observations, acquisition_window=acquisition_window,
            final_publication_horizon=final_publication_horizon, controls=solver_controls,
            limits=acquisition_limits, field_provider=provider, check=checkpoint,
            noise_provider=noise)
        phase = "electrical and readout execution"
        while runner.status not in ("physical_complete", "complete"):
            checkpoint()
            runner.advance_next()
        checkpoint()
        phase = "final digital publication"
        report = runner.finalize()
        terminal = "complete"
    except Exception as error:
        failure = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:1000],
                       executor_failure=None if runner is None or runner.failure is None else asdict(runner.failure))
    result = dict(base, status=terminal, failure=failure, elapsed_seconds=time.monotonic() - started,
        elapsed_limit_seconds=max_seconds,
        elapsed_limit_scope=("checked_between_operations_and_solver_iterations_not_hard_interrupt"
                             if array is not None and array.analog_conditioning is not None else
                             "checked_between_operations_not_inside_solver"),
        geometry=None if geometry is None else geometry.metadata,
        physical_noise=None if noise is None else dict(noise.metadata),
        hold=None if plan is None else dict(plan.report),
        physical_window=None if plan is None else (plan.intervals[0].start, plan.intervals[-1].end),
        covers_full_16_second_record=bool(plan is not None and plan.intervals[0].start == 0. and plan.intervals[-1].end == 16.),
        completed_readout=report is not None, readout=report,
        confirmed_physical_time=None if runner is None else runner.physical_time,
        electrical_state_advanced=bool(array is not None and array.completed_calls),
        accounting=None if array is None else array.accounting(),
        assembly_resources=None if array is None else asdict(array.resources),
        state_ids=() if array is None else array.state_ids,
        initial_state_volts=initial_state, initial_field_volts=initial_field,
        final_state_volts=None if array is None else array.volts,
        final_field_volts=None if array is None else array.previous_field_volts)
    return result


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"unsupported result item {type(value).__name__}")


def save_simulation_result(path, result, *, max_payload_bytes):
    """Write one exclusive HDF5 result container; never overwrite an attempt.

    The byte cap covers serialized strings and numeric payload, not HDF5 file
    metadata. Existing inputs/settings remain in their original retained files.
    Records are written in batches; no dense field or interval-state trajectory
    is created. An interrupted writer leaves record_complete=False, preserving
    the partial container. It is not a restart checkpoint.
    """
    import h5py
    _positive_int(max_payload_bytes, "max_payload_bytes")
    consumed = 0
    def charge(size):
        nonlocal consumed
        consumed += size
        if consumed > max_payload_bytes:
            raise ValueError("result payload exceeds its explicit byte cap")
    def encode(value):
        text = json.dumps(value, default=_json_default, sort_keys=True, allow_nan=False)
        charge(len(text.encode("utf-8")))
        return text
    string_type = h5py.string_dtype("utf-8")
    with h5py.File(Path(path), "x") as container:
        container.attrs["record_complete"] = False
        state_keys = ("initial_state_volts", "initial_field_volts", "final_state_volts", "final_field_volts")
        for key in state_keys:
            if result.get(key) is not None:
                values = np.asarray(result[key], dtype=np.float64)
                charge(values.nbytes)
                container.create_dataset(key, data=values)
        ids = result.get("state_ids", ())
        charge(sum(len(value.encode("utf-8")) for value in ids))
        container.create_dataset("state_ids", data=list(ids), dtype=string_type)
        readout = result.get("readout")
        sequences = {} if readout is None else {key: readout[key] for key in
                                              ("raw_values", "observations", "interval_evidence")}
        for key, records in sequences.items():
            target = container.create_dataset(key + "_json", shape=(len(records),), dtype=string_type)
            for start in range(0, len(records), 256):
                target[start:start + 256] = [encode(record) for record in records[start:start + 256]]
        metadata = {key: value for key, value in result.items() if key not in state_keys + ("state_ids", "readout")}
        metadata["readout"] = None if readout is None else {
            key: value for key, value in readout.items() if key not in sequences}
        container.create_dataset("metadata_json", data=encode(metadata))
        container.attrs["serialized_payload_bytes"] = consumed
        container.flush()
        container.attrs["record_complete"] = True
        container.flush()
