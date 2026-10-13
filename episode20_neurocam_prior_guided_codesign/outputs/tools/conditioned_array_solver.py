"""Joint conditional active-drive integration without changing passive topology.

Explicitly selected midpoint or Radau IIA and partition doubling supply consistency estimates,
not certified waveform/circuit-error bounds. All states, including contact and
ADC storage, advance together. The clipped controlled voltage is a target,
not a hard rail on a physical capacitor, which can be back-driven. No circuit
constants, stochastic noise, supply-power model or calibration are selected.
"""
from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from analog_conditioning import ControlledVoltageDrives
from conditioned_radau_step import radau_iia_step
from radau_contact_preconditioner import prepare_contact_blocks
from radau_triangular_contact_schur import prepare_triangular_contact_schur
from coupled_rc import NetworkInterval
from sparse_array_rc import (SolverControls, SolverConvergenceError, SparseInterval,
    SparsePassiveRCNetwork, PassiveStarConductance, _ConductanceWithStars,
    passive_matrix, _array, _charge_energy, _tuple)


@dataclass(frozen=True)
class ConditionedSolverControls(SolverControls):
    """Additional explicit iterative/work controls for the conditional model.

    max_linear_iterations and max_operator_products are shared across every
    attempted Newton solve and partition. Products count passive-G and active
    Jacobian applications, including residual/energy evaluations. C products,
    the existing sparse C factorization and its solves are outside that count.
    max_workspace_bytes bounds a conservative numeric-array workspace estimate;
    it does not certify SciPy/Python overhead or peak RSS. Caller check() and the
    enclosing allocation retain their deadline/RSS responsibilities.
    Either method permits up to 12 explicitly selected refinements;
    iteration/product/deadline caps still apply across the complete attempt.
    This does not change any caller's selected max_refinements or tolerances.
    Midpoint remains the default. Radau's numerical dissipation is separately
    reported and must satisfy the same original physical energy tolerance.
    """
    max_partition_refinements: ClassVar[int] = 12
    linear_relative_tolerance: float
    linear_charge_atol_coulomb: float
    energy_atol_joule: float
    max_newton_iterations: int
    max_linear_iterations: int
    gmres_restart: int
    max_workspace_bytes: int
    integration_method: str = "midpoint"
    radau_preconditioner: str = "stage_jacobi"
    radau_linear_solver: str = "gmres"

    def __post_init__(self):
        super().__post_init__()
        if self.integration_method not in ("midpoint", "radau_iia_2"):
            raise ValueError("integration_method must be midpoint or radau_iia_2")
        if self.radau_preconditioner not in ("stage_jacobi", "contact_block"):
            raise ValueError("radau_preconditioner must be stage_jacobi or contact_block")
        if self.radau_preconditioner == "contact_block" and self.integration_method != "radau_iia_2":
            raise ValueError("contact_block preconditioning requires radau_iia_2")
        if self.radau_linear_solver not in ("gmres", "triangular_contact_schur"):
            raise ValueError("radau_linear_solver must be gmres or triangular_contact_schur")
        if self.radau_linear_solver == "triangular_contact_schur" and self.integration_method != "radau_iia_2":
            raise ValueError("triangular_contact_schur requires radau_iia_2")
        for name in ("linear_relative_tolerance", "linear_charge_atol_coulomb", "energy_atol_joule"):
            value = getattr(self, name)
            if isinstance(value, bool) or np.iscomplexobj(value) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite, real and positive")
        for name in ("max_newton_iterations", "max_linear_iterations", "gmres_restart", "max_workspace_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class ConditionedInterval(SparseInterval):
    controlled_voltage_source_work_joule: float
    controlled_branch_dissipation_joule: float
    controlled_nodal_current_work_joule: float
    passive_norton_dissipation_joule: float
    controlled_buffer_shunt_quadrature_joule: float
    # Restores only the active buffer branch. Contact/external voltage-port
    # Norton equivalents still need the assembler's physical work restoration.
    controlled_port_corrected_norton_dissipation_joule: float
    energy_balance_residual_joule: float
    # Historical field name retained for compatibility. For Radau this is
    # the maximum accepted two-stage charge/voltage residual, not a midpoint.
    maximum_scaled_midpoint_residual: float
    newton_iterations: int
    linear_iterations: int
    estimated_work_error_joule: float
    time_varying_current_work_joule: float = 0.0
    physical_noise_work_joule: tuple = (0.0, 0.0, 0.0, 0.0)
    numerical_dissipation_joule: float = 0.0
    integration_method: str = "midpoint"
    radau_preconditioner: str = "stage_jacobi"
    radau_linear_solver: str = "gmres"
    radau_linear_solver_metadata: object = None


class _Counter:
    def __init__(self, controls, check):
        self.controls, self.check = controls, check
        self.products = self.newton = self.linear = 0

    def product(self):
        self.check()
        if self.products >= self.controls.max_operator_products:
            raise SolverConvergenceError("conditioned operator-product limit exceeded")
        self.products += 1

    def linear_step(self, _):
        self.check()
        self.linear += 1
        if self.linear > self.controls.max_linear_iterations:
            raise SolverConvergenceError("conditioned linear-iteration limit exceeded")


class ConditionedRCNetwork(SparsePassiveRCNetwork):
    """Same persistent C/state/preview ownership, with explicit active forcing."""

    def advance(self, *, duration_seconds, conductance_to_reference_siemens,
                resistor_edges, drive_current_amp, observation_matrix,
                feedthrough_volts, injected_charge_coulomb, controls,
                controlled_drives, check, passive_stars=None, commit=True,
                time_varying_forcing=None):
        """Propagate a held base drive with optional continuous local forcing.

        The frozen caller object evaluates ``evaluate(local_time_seconds)`` to
        ``(additional_current[N], ControlledVoltageDrives)``. Its time-specific
        amplifier input offset is applied before gain/clipping by that drive.
        ``work_rates(local_time_seconds, midpoint_state)`` returns four physical
        powers in fixed order: current-noise, access-voltage-noise, field-source
        correction, external-DAQ-source correction. These signed powers and
        the additional nodal-current power are integrated at every accepted
        internal midpoint, or both explicitly selected Radau stages, and all
        participate in partition refinement. Radau additionally requires
        evaluate_left/work_rates_left for the current interval's left-limit
        endpoint; it never samples the next physical switch/charge interval.

        Numerical refinement re-evaluates the SAME frozen forcing path, never
        redraws or resets it. The adapter owns exact interval support, original
        physical ports and coefficient storage; this solver owns state/work
        quadrature and its workspace. None retains the existing held path.
        """
        if not isinstance(controls, ConditionedSolverControls) or not callable(check):
            raise ValueError("explicit conditioned controls and resource callback required")
        if not isinstance(controlled_drives, ControlledVoltageDrives) or controlled_drives.state_count != self._size:
            raise ValueError("bound controlled drives must match the entire state basis")
        if time_varying_forcing is not None and (not callable(getattr(time_varying_forcing, "evaluate", None))
                or not callable(getattr(time_varying_forcing, "work_rates", None))):
            raise ValueError("time-varying forcing requires explicit evaluation and physical work methods")
        use_radau = controls.integration_method == "radau_iia_2"
        if (use_radau and time_varying_forcing is not None
                and any(not callable(getattr(time_varying_forcing, name, None))
                        for name in ("evaluate_left", "work_rates_left"))):
            raise ValueError("Radau forcing requires current-interval left-limit methods")
        if type(commit) is not bool:
            raise ValueError("commit must be boolean")
        if (isinstance(duration_seconds, bool) or np.iscomplexobj(duration_seconds)
                or not math.isfinite(duration_seconds) or duration_seconds <= 0):
            raise ValueError("positive finite duration required")
        check()
        self._pending_preview = None
        duration, size = float(duration_seconds), self._size
        restart = min(controls.gmres_restart, 2 * size if use_radau else size)
        if use_radau:
            # This is the helper's current full 2N numeric bound, PLUS the
            # simultaneously retained outer vectors. Both share ONE cap.
            # Keep the helper term synchronized with conditioned_radau_step;
            # its repeated admission is not a second allocation allowance.
            workspace = 8 * ((restart + 80) * (2 * size) + (restart + 1) * restart
                             + 12 * restart + 14 * 64 + 20)
            if time_varying_forcing is not None:
                workspace += 8 * (6 * size + 2 * 64)
            workspace += 8 * (32 * size + 40)
        else:
            workspace = 8 * ((restart + 40) * size + (restart + 1) * restart + 8 * restart)
            if time_varying_forcing is not None:
                # Owned additional current and combined base current, plus the
                # evaluated caller vector while it is copied. Coefficient storage
                # belongs to the separately bounded adapter, not this workspace.
                workspace += 8 * (3 * size + 10)
        if workspace > controls.max_workspace_bytes:
            raise SolverConvergenceError("conditioned numeric-workspace limit exceeded")
        shunts = _array("conductance_to_reference_siemens", conductance_to_reference_siemens, (size,))
        nodes = np.asarray(controlled_drives.buffer_nodes, dtype=np.int64)
        # +gb*b damping is the passive half of the declared gb*(u-b) branch.
        # Additional explicitly supplied shunts remain independent passive loads.
        required_shunts = np.asarray(controlled_drives.buffer_conductance_siemens, dtype=float)
        if np.any(shunts[nodes] < required_shunts):
            raise ValueError("buffer shunts must include every controlled branch conductance")
        conductance = passive_matrix(shunts=shunts, edges=resistor_edges, positive_shunts=False)
        if passive_stars is not None:
            if not isinstance(passive_stars, PassiveStarConductance) or passive_stars.shape != conductance.shape:
                raise ValueError("validated passive stars in the same basis required")
            conductance = _ConductanceWithStars(conductance, passive_stars)
        current = _array("drive_current_amp", drive_current_amp, (size,))
        injected = _array("injected_charge_coulomb", injected_charge_coulomb, (size,))
        if np.iscomplexobj(observation_matrix):
            raise ValueError("real observation matrix required")
        observation = csr_matrix(observation_matrix, dtype=float, copy=True)
        if (observation.shape[1] != size or observation.shape[0] < 1
                or not np.all(np.isfinite(observation.data))):
            raise ValueError("finite observation matrix in full state basis required")
        if use_radau:
            # Observation temporaries join the same combined outer/helper
            # budget. Sparse topology, C factor and forcing coefficients are
            # already admitted separately; Python overhead remains RSS-bound.
            workspace += 8 * (8 * observation.shape[0])
            if workspace > controls.max_workspace_bytes:
                raise SolverConvergenceError("Radau combined numeric-workspace limit exceeded")
        triangular_plan, triangular_info = None, {}
        if use_radau and controls.radau_linear_solver == "triangular_contact_schur":
            # Preparation inspects the actual interval matrices and admits its
            # complete additional workspace before any boundary transition.
            # Only unsupported structure may fall back; failures propagate.
            triangular_plan = prepare_triangular_contact_schur(self._capacitance,
                conductance, controlled_drives,
                max_workspace_bytes=controls.max_workspace_bytes - workspace,
                check=check, fallback_info=triangular_info)
        contact_block_plan = None
        if use_radau and triangular_plan is None and controls.radau_preconditioner == "contact_block":
            # All additional preparation, factors and apply temporaries share
            # the SAME outer/helper cap; the plan receives only its remainder.
            # Unsupported topology retains the existing stage-Jacobi method.
            contact_block_plan = prepare_contact_blocks(self._capacitance, conductance,
                max_workspace_bytes=controls.max_workspace_bytes - workspace, check=check)
        feedthrough = _array("feedthrough_volts", feedthrough_volts, (observation.shape[0],))
        start, boundary = self._transition(self._capacitance, self._factor, injected, controls)
        counter = _Counter(controls, check)
        capacitance = self._capacitance
        radau_diagonal = conductance.diagonal() if use_radau else None

        def passive_product(value):
            counter.product()
            result = conductance @ value
            if not np.all(np.isfinite(result)):
                raise SolverConvergenceError("nonfinite passive product")
            return result

        def voltage_error(left, right):
            scale = controls.voltage_atol + controls.relative_tolerance * np.maximum(np.abs(left), np.abs(right))
            return float(np.max(np.abs(left - right) / scale))

        def midpoint(old, step, local_time):
            # Solving for the endpoint keeps the C charge equation explicit.
            drives, step_current, additional = controlled_drives, current, None
            if time_varying_forcing is not None:
                check()
                additional, drives = time_varying_forcing.evaluate(local_time)
                additional = _array("time-varying current", additional, (size,))
                if (not isinstance(drives, ControlledVoltageDrives) or drives.state_count != size
                        or drives.source_nodes != controlled_drives.source_nodes
                        or drives.buffer_nodes != controlled_drives.buffer_nodes
                        or drives.buffer_conductance_siemens != controlled_drives.buffer_conductance_siemens):
                    raise ValueError("time-varying controlled drive must retain its full basis and branch conductances")
                with np.errstate(over="raise", invalid="raise"):
                    try:
                        step_current = current + additional
                    except FloatingPointError as error:
                        raise SolverConvergenceError("nonfinite combined time-varying current") from error
                if not np.all(np.isfinite(step_current)):
                    raise SolverConvergenceError("nonfinite combined time-varying current")
            endpoint = old.copy()
            preconditioner_diagonal = capacitance.diagonal() + 0.5 * step * conductance.diagonal()
            if not np.all(np.isfinite(preconditioner_diagonal)) or np.any(preconditioner_diagonal <= 0):
                raise SolverConvergenceError("invalid midpoint preconditioner")
            preconditioner = LinearOperator((size, size), matvec=lambda x: x / preconditioner_diagonal,
                                            dtype=np.dtype(float))
            for _ in range(controls.max_newton_iterations):
                check()
                counter.newton += 1
                middle = 0.5 * old + 0.5 * endpoint
                active = drives.evaluate(middle)
                if tuple(active.node_indices) != tuple(controlled_drives.buffer_nodes):
                    raise ValueError("controlled drive node order changed")
                forcing = step_current.copy()
                np.add.at(forcing, nodes, active.current_amp)
                passive = passive_product(middle)
                charge_change = capacitance @ (endpoint - old)
                right_charge = step * (forcing - passive)
                residual = charge_change - right_charge
                denominator = controls.charge_atol_coulomb + controls.relative_tolerance * np.maximum(
                    np.abs(charge_change), np.abs(right_charge))
                charge_error = float(np.max(np.abs(residual) / denominator))
                correction_in_volts = self._factor.solve(residual)
                voltage_scale = controls.voltage_atol + controls.relative_tolerance * np.maximum(np.abs(old), np.abs(endpoint))
                residual_error = max(charge_error, float(np.max(np.abs(correction_in_volts) / voltage_scale)))
                if not math.isfinite(residual_error):
                    raise SolverConvergenceError("nonfinite midpoint residual")
                if residual_error <= 1:
                    powers = np.array([active.controlled_voltage_source_power_watt,
                        active.controlled_branch_dissipation_watt, active.controlled_nodal_current_power_watt,
                        float(middle @ passive), active.buffer_shunt_power_watt], dtype=float)
                    if time_varying_forcing is not None:
                        check()
                        physical = _array("physical noise work rates",
                            time_varying_forcing.work_rates(local_time, middle), (4,))
                        powers = np.concatenate((powers, [float(additional @ middle)], physical))
                    if not np.all(np.isfinite(powers)):
                        raise SolverConvergenceError("nonfinite midpoint work")
                    return endpoint, middle, step * powers, residual_error

                def jacobian_product(direction):
                    result = capacitance @ direction + 0.5 * step * passive_product(direction)
                    counter.product()
                    derivative = drives.jacobian_product(middle, direction)
                    np.add.at(result, nodes, -0.5 * step * derivative)
                    if not np.all(np.isfinite(result)):
                        raise SolverConvergenceError("nonfinite midpoint Jacobian product")
                    return result
                remaining = controls.max_linear_iterations - counter.linear
                if remaining <= 0:
                    raise SolverConvergenceError("conditioned linear-iteration limit exceeded")
                jacobian = LinearOperator((size, size), matvec=jacobian_product, dtype=np.dtype(float))
                delta, info = gmres(jacobian, -residual, M=preconditioner,
                    rtol=controls.linear_relative_tolerance, atol=controls.linear_charge_atol_coulomb,
                    restart=restart, maxiter=remaining, callback=counter.linear_step, callback_type="legacy")
                if info != 0 or not np.all(np.isfinite(delta)):
                    raise SolverConvergenceError("conditioned GMRES did not converge")
                endpoint += delta
                if not np.all(np.isfinite(endpoint)):
                    raise SolverConvergenceError("nonfinite midpoint endpoint")
            raise SolverConvergenceError("conditioned Newton limit exceeded")

        def propagate(parts):
            voltage = start.copy()
            mean, works = np.zeros(size), np.zeros(5 if time_varying_forcing is None else 10)
            maximum_residual = 0.0
            numerical_dissipation = 0.0
            step = duration / parts
            if not math.isfinite(step) or step <= 0:
                raise SolverConvergenceError("unrepresentable integration step")
            for index in range(parts):
                check()
                if use_radau:
                    left = duration * (index / parts)
                    right = duration if index + 1 == parts else duration * ((index + 1) / parts)
                    local_step = right - left
                    first_stage = left + local_step / 3.
                    if not (0.0 <= left < first_stage < right <= duration):
                        raise SolverConvergenceError("unrepresentable internal Radau stage times")
                    step_result = radau_iia_step(capacitance=capacitance,
                        factor_solve=self._factor.solve, passive_product=passive_product,
                        passive_diagonal=radau_diagonal, controls=controls,
                        counter=counter, check=check, old=voltage, current=current,
                        controlled_drives=controlled_drives, step_seconds=local_step,
                        stage_times=(first_stage, right), time_varying_forcing=time_varying_forcing,
                        contact_block_plan=contact_block_plan, triangular_plan=triangular_plan)
                    voltage, middle, increments = step_result.endpoint, step_result.mean, step_result.work_increments
                    residual_error = step_result.maximum_scaled_residual
                    numerical_dissipation += step_result.numerical_dissipation_joule
                    mean += (local_step / duration) * middle
                else:
                    local_time = None if time_varying_forcing is None else duration * ((index + 0.5) / parts)
                    if time_varying_forcing is not None and not 0.0 < local_time < duration:
                        raise SolverConvergenceError("unrepresentable internal midpoint time")
                    voltage, middle, increments, residual_error = midpoint(voltage, step, local_time)
                    mean += middle / parts
                works += increments
                maximum_residual = max(maximum_residual, residual_error)
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(works)):
                raise SolverConvergenceError("nonfinite accumulated interval quantities")
            if not math.isfinite(numerical_dissipation) or numerical_dissipation < 0:
                raise SolverConvergenceError("invalid accumulated numerical dissipation")
            return voltage, mean, works, maximum_residual, numerical_dissipation

        coarse_end, coarse_mean, coarse_work, _, _ = propagate(1)
        failure_reason = "conditioned partition agreement exceeds declared tolerance"
        for level in range(1, controls.max_refinements + 1):
            parts = 2**level
            end, mean, work, maximum_residual, numerical_dissipation = propagate(parts)
            failure_reason = "conditioned partition agreement exceeds declared tolerance"
            output_end, output_mean = observation @ end + feedthrough, observation @ mean + feedthrough
            work_scale = controls.energy_atol_joule + controls.relative_tolerance * np.maximum(np.abs(work), np.abs(coarse_work))
            error = max(voltage_error(end, coarse_end), voltage_error(mean, coarse_mean),
                        voltage_error(output_end, observation @ coarse_end + feedthrough),
                        voltage_error(output_mean, observation @ coarse_mean + feedthrough),
                        float(np.max(np.abs(work - coarse_work) / work_scale)))
            if math.isfinite(error) and error <= 1:
                start_charge, start_energy = _charge_energy(capacitance, start)
                end_charge, end_energy = _charge_energy(capacitance, end)
                base_work = float(duration * (current @ mean))
                source_work, branch_heat, nodal_work, passive_heat, buffer_shunt = map(float, work[:5])
                varying_work = 0.0 if time_varying_forcing is None else float(work[5])
                physical_work = ((0.0, 0.0, 0.0, 0.0) if time_varying_forcing is None
                                 else tuple(float(value) for value in work[6:10]))
                corrected_norton_heat = passive_heat - buffer_shunt + branch_heat
                residual = float(end_energy - start_energy - base_work - nodal_work + passive_heat)
                if time_varying_forcing is not None:
                    residual -= varying_work
                energy_scale = controls.energy_atol_joule + controls.relative_tolerance * max(
                    abs(start_energy), abs(end_energy), abs(base_work), abs(nodal_work), abs(passive_heat),
                    abs(varying_work))
                if use_radau:
                    corrected_residual = residual + numerical_dissipation
                    if not all(math.isfinite(x) for x in (base_work, corrected_norton_heat, residual,
                                                         energy_scale, corrected_residual)):
                        raise SolverConvergenceError("nonfinite Radau energy bookkeeping")
                    # Do not conceal numerical energy loss inside physical
                    # heat or accept it by compensation. The unchanged raw
                    # physical balance and Dnum BOTH use the original scale;
                    # the corrected RK identity checks stage bookkeeping.
                    if max(abs(residual), numerical_dissipation, abs(corrected_residual)) > energy_scale:
                        failure_reason = "Radau energy balance or numerical dissipation exceeds declared tolerance"
                        coarse_end, coarse_mean, coarse_work = end, mean, work
                        continue
                elif (not all(math.isfinite(x) for x in (base_work, corrected_norton_heat, residual, energy_scale))
                        or abs(residual) > energy_scale):
                    raise SolverConvergenceError("midpoint energy identity exceeds declared tolerance")
                input_work = base_work + nodal_work
                if time_varying_forcing is not None:
                    input_work += varying_work
                    if not math.isfinite(input_work):
                        raise SolverConvergenceError("nonfinite combined interval input work")
                interval = NetworkInterval(_tuple(start), _tuple(end), _tuple(mean), _tuple(output_end),
                    _tuple(output_mean), _tuple(start_charge), _tuple(end_charge), start_energy, end_energy,
                    input_work, duration)
                result = ConditionedInterval(boundary, interval,
                    float(np.max(np.abs(end - coarse_end))), float(np.max(np.abs(mean - coarse_mean))),
                    error, parts, counter.products, source_work, branch_heat, nodal_work, passive_heat,
                    buffer_shunt, corrected_norton_heat, residual, maximum_residual, counter.newton, counter.linear,
                    float(np.max(np.abs(work - coarse_work))), varying_work, physical_work,
                    numerical_dissipation, controls.integration_method,
                    ("not_used" if triangular_plan is not None else
                     "contact_block" if contact_block_plan is not None else "stage_jacobi"),
                    "triangular_contact_schur" if triangular_plan is not None else "gmres",
                    (dict(triangular_info, **triangular_plan.metadata)
                     if triangular_plan is not None else dict(triangular_info)))
                check()
                if commit:
                    self._volts = end.copy()
                else:
                    self._pending_preview = result
                return result
            coarse_end, coarse_mean, coarse_work = end, mean, work
        raise SolverConvergenceError(failure_reason)
