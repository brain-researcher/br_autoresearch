"""One bounded two-stage Radau IIA step for the existing full circuit.

This internal helper owns no persistent state, topology, boundary transitions,
outer refinement or acceptance decision. The caller supplies the already
admitted capacitance factor, counted passive product and shared resource
counter. Every stage uses the same original physical forcing and work ports.
The second stage queries their explicit left-limit interface, never the next
physical interval. Numerical dissipation is returned separately from heat.
No production method, tolerance, physical constant or resource cap is chosen.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from analog_conditioning import ControlledVoltageDrives
from sparse_array_rc import SolverConvergenceError, _array


@dataclass(frozen=True)
class RadauStep:
    endpoint: np.ndarray
    mean: np.ndarray
    work_increments: np.ndarray
    maximum_scaled_residual: float
    numerical_dissipation_joule: float
    estimated_workspace_bytes: int


def radau_iia_step(*, capacitance, factor_solve, passive_product, passive_diagonal,
                   controls, counter, check, old, current, controlled_drives,
                   step_seconds, stage_times, time_varying_forcing=None,
                   contact_block_plan=None, triangular_plan=None):
    """Return one preview using A=((5/12,-1/12),(3/4,1/4)), b=(3/4,1/4).

    ``controls`` and ``counter`` are the existing ConditionedSolverControls
    and shared _Counter interfaces; avoiding their import prevents a cycle
    when the outer solver imports this helper. ``passive_product`` already
    charges every G application. Active Jacobian products are charged here.
    C products/factor solves retain the existing counter convention.

    The caller supplies local stage times (t+h/3,t+h). Optional forcing must
    implement evaluate/work_rates and evaluate_left/work_rates_left. The
    latter uses the current interval's coefficients/configuration at its end.
    Returned work order is the existing five, or ten with continuous forcing.
    The returned nonnegative numerical dissipation is NOT physical heat and
    must not replace the outer solver's original physical balance tolerance.
    """
    if not callable(check) or not callable(factor_solve) or not callable(passive_product):
        raise ValueError("Radau requires the existing checked full-state operators")
    if not isinstance(controlled_drives, ControlledVoltageDrives):
        raise ValueError("Radau requires the bound controlled drives")
    size = controlled_drives.state_count
    if capacitance.shape != (size, size):
        raise ValueError("Radau capacitance must retain the complete state basis")
    if (isinstance(step_seconds, (bool, np.bool_)) or not math.isfinite(step_seconds)
            or step_seconds <= 0):
        raise ValueError("positive finite Radau step required")
    if (not isinstance(stage_times, (tuple, list)) or len(stage_times) != 2
            or any(isinstance(t, (bool, np.bool_)) or not math.isfinite(t) for t in stage_times)
            or not 0 <= stage_times[0] < stage_times[1]):
        raise ValueError("two ordered finite local Radau stage times required")
    if time_varying_forcing is not None and any(not callable(getattr(time_varying_forcing, name, None))
            for name in ("evaluate", "evaluate_left", "work_rates", "work_rates_left")):
        raise ValueError("Radau forcing requires explicit left-limit evaluation and physical work")
    check()
    block_size = 2 * size
    restart = min(controls.gmres_restart, block_size)
    # Conservative live numeric storage: 2N Arnoldi basis and work vectors,
    # both stages/residuals/Jacobian products, preconditioner, copied inputs,
    # and two forcing vectors/views. Matrix/factor/caller coefficient storage
    # is separately admitted. This is not a claim about Python/SciPy peak RSS.
    workspace = 8 * ((restart + 80) * block_size + (restart + 1) * restart
                     + 12 * restart + 14 * 64 + 20)
    if time_varying_forcing is not None:
        workspace += 8 * (6 * size + 2 * 64)
    if contact_block_plan is not None:
        extra = contact_block_plan.estimated_workspace_bytes
        if (contact_block_plan.state_count != size or type(extra) is not int or extra < 0):
            raise ValueError("contact plan must retain the complete admitted state basis")
        workspace += extra
    if triangular_plan is not None:
        if contact_block_plan is not None:
            raise ValueError("direct Radau solve cannot also select a preconditioner")
        extra = triangular_plan.estimated_workspace_bytes
        if triangular_plan.state_count != size or type(extra) is not int or extra < 0:
            raise ValueError("triangular plan must retain the complete admitted state basis")
        workspace += extra
    if workspace > controls.max_workspace_bytes:
        raise SolverConvergenceError("Radau 2N numeric-workspace limit exceeded")

    step = float(step_seconds)
    old = _array("Radau initial state", old, (size,))
    current = _array("Radau held current", current, (size,))
    diagonal = _array("Radau passive diagonal", passive_diagonal, (size,))
    cdiagonal = _array("Radau capacitance diagonal", capacitance.diagonal(), (size,))
    if np.any(cdiagonal <= 0) or np.any(diagonal < 0):
        raise ValueError("Radau requires positive capacitance and nonnegative passive diagonal")
    nodes = np.asarray(controlled_drives.buffer_nodes, dtype=np.int64)
    drives, currents, additional = [], [], []
    for index, point in enumerate(stage_times):
        check()
        active, extra = controlled_drives, None
        if time_varying_forcing is not None:
            evaluate = time_varying_forcing.evaluate if index == 0 else time_varying_forcing.evaluate_left
            extra, active = evaluate(float(point))
            extra = _array("Radau time-varying current", extra, (size,))
            if (not isinstance(active, ControlledVoltageDrives) or active.state_count != size
                    or active.source_nodes != controlled_drives.source_nodes
                    or active.buffer_nodes != controlled_drives.buffer_nodes
                    or active.buffer_conductance_siemens != controlled_drives.buffer_conductance_siemens):
                raise ValueError("Radau forcing must retain the controlled full-state basis and conductances")
        total = current.copy() if extra is None else current + extra
        if not np.all(np.isfinite(total)):
            raise SolverConvergenceError("nonfinite Radau combined current")
        drives.append(active)
        currents.append(total)
        additional.append(extra)

    if triangular_plan is not None:
        # The two forcing snapshots above are consumed exactly once. Every
        # recovered original stage still passes the unmodified residual/work
        # gate below; no iterative solver or preconditioner is constructed.
        stages = triangular_plan.solve_stages(step_seconds=step, old=old,
            stage_currents=tuple(currents), stage_drives=tuple(drives),
            passive_product=passive_product, check=check)
        stages = _array("recovered triangular Radau stages", stages, (2, size))
    else:
        # Nodewise 2x2 stage-Jacobi inverse. It retains the Radau stage coupling
        # without forming a dense 2N matrix; controlled cross-node terms remain
        # in the actual Jacobian. Scaling avoids squaring dimensional C/G values.
        q = step * diagonal
        scale = np.maximum(cdiagonal, q)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(scale)):
            raise SolverConvergenceError("nonfinite Radau preconditioner")
        c, g = cdiagonal / scale, q / scale
        determinant = scale * (c * c + (2. / 3.) * c * g + g * g / 6.)
        if np.any(determinant <= 0) or not np.all(np.isfinite(determinant)):
            raise SolverConvergenceError("invalid Radau preconditioner")
        p00, p01 = (c + g / 4.) / determinant, (g / 12.) / determinant
        p10, p11 = (-3. * g / 4.) / determinant, (c + 5. * g / 12.) / determinant
        if any(not np.all(np.isfinite(item)) for item in (p00, p01, p10, p11)):
            raise SolverConvergenceError("unrepresentable Radau preconditioner")
    
        def precondition(value):
            check()
            first, second = value.reshape(2, size)
            return np.concatenate((p00 * first + p01 * second, p10 * first + p11 * second))
    
        preconditioner = LinearOperator((block_size, block_size), matvec=precondition, dtype=np.dtype(float))
        if contact_block_plan is not None:
            # Prepared by the caller from this interval's actual C/G topology and
            # admitted against the combined outer/helper numeric-workspace cap.
            # Only GMRES M changes; the physical residual and Jacobian below do not.
            preconditioner = contact_block_plan.for_step(step, fallback=preconditioner, check=check)
        stages = np.vstack((old, old))
    for _ in range(controls.max_newton_iterations):
        check()
        if triangular_plan is None:
            counter.newton += 1
        evaluations, passive_values, forces = [], [], []
        for index in range(2):
            check()
            active = drives[index].evaluate(stages[index])
            if tuple(active.node_indices) != tuple(controlled_drives.buffer_nodes):
                raise ValueError("Radau controlled node order changed")
            passive = _array("Radau passive product", passive_product(stages[index]), (size,))
            forcing = currents[index].copy()
            np.add.at(forcing, nodes, active.current_amp)
            force = forcing - passive
            if not np.all(np.isfinite(force)):
                raise SolverConvergenceError("nonfinite Radau stage force")
            evaluations.append(active)
            passive_values.append(passive)
            forces.append(force)
        change = stages - old
        charge = np.vstack((capacitance @ change[0], capacitance @ change[1]))
        right = step * np.vstack(((5. * forces[0] - forces[1]) / 12.,
                                 (3. * forces[0] + forces[1]) / 4.))
        residual = charge - right
        denominator = controls.charge_atol_coulomb + controls.relative_tolerance * np.maximum(np.abs(charge), np.abs(right))
        correction = np.vstack((factor_solve(residual[0]), factor_solve(residual[1])))
        voltage_scale = controls.voltage_atol + controls.relative_tolerance * np.maximum(np.abs(old), np.abs(stages))
        error = max(float(np.max(np.abs(residual) / denominator)),
                    float(np.max(np.abs(correction) / voltage_scale)))
        if (not np.all(np.isfinite(residual)) or not np.all(np.isfinite(correction))
                or not math.isfinite(error)):
            raise SolverConvergenceError("nonfinite Radau stage residual")
        if error <= 1:
            powers = []
            for index, point in enumerate(stage_times):
                check()
                active, state = evaluations[index], stages[index]
                values = np.array([active.controlled_voltage_source_power_watt,
                    active.controlled_branch_dissipation_watt, active.controlled_nodal_current_power_watt,
                    float(state @ passive_values[index]), active.buffer_shunt_power_watt])
                if time_varying_forcing is not None:
                    work_rates = time_varying_forcing.work_rates if index == 0 else time_varying_forcing.work_rates_left
                    physical = _array("Radau original physical work rates", work_rates(float(point), state), (4,))
                    values = np.concatenate((values, [float(additional[index] @ state)], physical))
                if not np.all(np.isfinite(values)):
                    raise SolverConvergenceError("nonfinite Radau physical power")
                powers.append(values)
            endpoint = stages[1].copy()
            mean = .75 * stages[0] + .25 * stages[1]
            increments = step * (.75 * powers[0] + .25 * powers[1])
            # h*(k1-k2)=2*(3*(Y1-old)-(Y2-old)); no C inverse is needed.
            defect = 3. * change[0] - change[1]
            dissipation = float(.125 * (defect @ (capacitance @ defect)))
            if (not all(np.all(np.isfinite(value)) for value in (endpoint, mean, increments))
                    or not math.isfinite(dissipation) or dissipation < 0):
                raise SolverConvergenceError("nonfinite Radau step or numerical dissipation")
            check()
            for value in (endpoint, mean, increments):
                value.flags.writeable = False
            return RadauStep(endpoint, mean, increments, error, dissipation, workspace)

        if triangular_plan is not None:
            # Direct stage recovery is accepted only by the original full
            # componentwise charge and C-inverse voltage residual above.
            # Numerical failure never switches methods or relaxes tolerances.
            raise SolverConvergenceError("triangular Radau recovered stage residual exceeds original tolerance")

        def jacobian_product(value):
            direction = value.reshape(2, size)
            derivatives = []
            for index in range(2):
                change_force = -_array("Radau passive Jacobian product", passive_product(direction[index]), (size,))
                counter.product()
                active_change = drives[index].jacobian_product(stages[index], direction[index])
                np.add.at(change_force, nodes, active_change)
                derivatives.append(change_force)
            result = np.concatenate((capacitance @ direction[0] - step * (5. * derivatives[0] - derivatives[1]) / 12.,
                                     capacitance @ direction[1] - step * (3. * derivatives[0] + derivatives[1]) / 4.))
            if not np.all(np.isfinite(result)):
                raise SolverConvergenceError("nonfinite Radau Jacobian product")
            return result

        remaining = controls.max_linear_iterations - counter.linear
        if remaining <= 0:
            raise SolverConvergenceError("conditioned linear-iteration limit exceeded")
        jacobian = LinearOperator((block_size, block_size), matvec=jacobian_product, dtype=np.dtype(float))
        delta, info = gmres(jacobian, -residual.ravel(), M=preconditioner,
            rtol=controls.linear_relative_tolerance, atol=controls.linear_charge_atol_coulomb,
            restart=restart, maxiter=remaining, callback=counter.linear_step, callback_type="legacy")
        if info != 0 or not np.all(np.isfinite(delta)):
            raise SolverConvergenceError("Radau GMRES did not converge")
        stages += delta.reshape(2, size)
        if not np.all(np.isfinite(stages)):
            raise SolverConvergenceError("nonfinite Radau stage state")
    raise SolverConvergenceError("Radau Newton limit exceeded")
