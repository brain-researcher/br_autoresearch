"""Finite full-equation checks for exact contact-state elimination.

This file contains no saved-world or device outcome inputs.  Execution belongs
in the coordinator's Slurm allocation.  Runtime integration checks require
the separately prepared integration patch; helper checks do not imply
that the production runner has selected this solver.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from analog_conditioning import ControlledVoltageDrives
from conditioned_array_solver import ConditionedRCNetwork
from radau_triangular_contact_schur import prepare_triangular_contact_schur
from sparse_array_rc import PassiveStarConductance, SolverConvergenceError, _ConductanceWithStars
from test_conditioned_array_solver import channel, controls


A = np.array([[5. / 12., -1. / 12.], [.75, .25]])
SOURCES = np.arange(2, 66)
CONTACTS = np.arange(66, 98)
BUFFERS = np.arange(98, 162)
FILTERS = np.arange(162, 226)
HOLDS = np.arange(226, 228)
SIZE = 228


def _edge(matrix, first, second, coefficient):
    # Independent nodal edge stamp, with no solver extraction/permutation code.
    matrix[first, first] += coefficient
    matrix[second, second] += coefficient
    matrix[first, second] -= coefficient
    matrix[second, first] -= coefficient


def finite_problem(*, clipped=False, cancellation=False):
    ground = np.r_[.7, .9, np.full(64, 1.1), np.zeros(32),
                   np.full(64, .5), np.full(64, 1.4), np.full(2, .3)]
    if cancellation:
        ground[:2] = 1e-3
    c = np.diag(ground)
    cap_edges = []
    groups = []
    for pad in range(2):
        nodes = (pad, *range(66 + 16 * pad, 82 + 16 * pad))
        branch_cap = np.linspace(.03, .14, 16) * (1e4 if cancellation else 1.)
        for node, value in zip(nodes[1:], branch_cap):
            cap_edges.append((pad, node, float(value)))
        weights = np.r_[.02 + .01 * pad, np.linspace(.1, .7, 16)]
        if cancellation:
            weights *= 1e4
        groups.append((nodes, tuple(weights), 1e4 if cancellation else .3 + .2 * pad))
    cap_edges.append((0, 1, .015))
    cap_edges.extend((int(first), int(first + 1), .02) for first in SOURCES[:-1])
    for edge in cap_edges:
        _edge(c, *edge)
    shunts = np.r_[.03, .04, np.full(64, .2), np.zeros(32),
                   np.full(64, 2.), np.full(64, .1), np.full(2, .03)]
    edges = [(0, 2, .4), (1, 3, .6)]
    edges.extend((int(first), int(second), .7) for first, second in zip(BUFFERS, FILTERS))
    edges.extend(((162, 226, .8), (163, 226, .01), (163, 227, .9), (162, 227, .015)))
    base = np.diag(shunts)
    for edge in edges:
        _edge(base, *edge)
    stars = PassiveStarConductance(state_count=SIZE, groups=groups)
    operator = _ConductanceWithStars(csr_matrix(base), stars)
    # The oracle uses the original full operator, never the Schur helper's
    # local blocks, group signatures, coordinate transform or band assembly.
    g = base + stars @ np.eye(SIZE)
    channels = tuple(channel(gain=0. if cancellation else 1.2, offset_volts=0.,
        drive_lower_volts=-.2 if clipped and index == 0 else -10.,
        drive_upper_volts=.2 if clipped and index == 0 else 10.) for index in range(64))
    drive = ControlledVoltageDrives(channels=channels, source_nodes=tuple(map(int, SOURCES)),
        buffer_nodes=tuple(map(int, BUFFERS)), state_count=SIZE)
    noises = (.003 * np.sin(np.arange(64)), -.004 * np.cos(np.arange(64)))
    if clipped:
        noises[0][0], noises[1][0] = .5, -.5
    stage_drives = tuple(drive.with_input_noise_volts(noise) for noise in noises)
    initial = .012 * np.sin(.3 + np.arange(SIZE))
    # The step helper receives the complete post-boundary-charge state.
    injected = np.zeros(SIZE)
    injected[0], injected[66], injected[227] = .003, -.002, .001
    old = initial + np.linalg.solve(c, injected)
    currents = np.vstack((.011 * np.sin(np.arange(SIZE) + .4),
                          .009 * np.cos(np.arange(SIZE) - .2)))
    return dict(c=c, sparse_c=csr_matrix(c), g=g, operator=operator, base=base,
        stars=stars, groups=groups, channels=channels, drive=drive,
        stage_drives=stage_drives, noises=noises, old=old, initial=initial,
        injected=injected, currents=currents, ground=ground, cap_edges=cap_edges,
        shunts=shunts, edges=edges)


def dense_stage_oracle(problem, step, *, clipped=False):
    """Solve all original physical coordinates in one dense 2N system.

    Only channel0 is deliberately clipped, to opposite rails in the two
    stages. Its known masks are verified against the resulting source states.
    Other channels remain affine. No elimination is used in this reference.
    """
    c, g, old = (problem[name] for name in ("c", "g", "old"))
    matrices, offsets = [], []
    for stage in range(2):
        active = np.zeros_like(c)
        offset = np.zeros(SIZE)
        for index, spec in enumerate(problem["channels"]):
            if clipped and index == 0:
                target = spec.drive_upper_volts if stage == 0 else spec.drive_lower_volts
                offset[BUFFERS[index]] = spec.buffer_conductance_siemens * target
            else:
                active[BUFFERS[index], SOURCES[index]] = spec.buffer_conductance_siemens * spec.gain
                offset[BUFFERS[index]] = spec.buffer_conductance_siemens * (
                    spec.offset_volts + spec.gain * problem["noises"][stage][index])
        matrices.append(active)
        offsets.append(offset)
    full = np.block([[((c if first == second else np.zeros_like(c))
        + step * A[first, second] * (g - matrices[second])) for second in range(2)]
        for first in range(2)])
    rhs = np.vstack((c @ old, c @ old)) + step * A @ (problem["currents"] + offsets)
    result = np.linalg.solve(full, rhs.ravel()).reshape(2, SIZE)
    for stage in range(2):
        requested = np.array([spec.gain * (result[stage, SOURCES[index]]
            + problem["noises"][stage][index]) + spec.offset_volts
            for index, spec in enumerate(problem["channels"])])
        if clipped:
            if stage == 0 and not requested[0] > problem["channels"][0].drive_upper_volts:
                raise AssertionError("the first-stage oracle was not upper clipped")
            if stage == 1 and not requested[0] < problem["channels"][0].drive_lower_volts:
                raise AssertionError("the second-stage oracle was not lower clipped")
        start = 1 if clipped else 0
        if any(not spec.drive_lower_volts < requested[index] < spec.drive_upper_volts
               for index, spec in enumerate(problem["channels"]) if index >= start):
            raise AssertionError("an oracle affine channel unexpectedly clipped")
    return result


def full_residual(problem, step, stages):
    forces = []
    for index in range(2):
        active = problem["stage_drives"][index].evaluate(stages[index])
        force = problem["currents"][index] - problem["operator"] @ stages[index]
        force[BUFFERS] += active.current_amp
        forces.append(force)
    charge = (problem["c"] @ (stages - problem["old"]).T).T
    right = step * A @ np.asarray(forces)
    return charge - right, charge, right


def prepare(problem, **changes):
    arguments = dict(max_workspace_bytes=32 * 1024**2, check=lambda: None)
    arguments.update(changes)
    return prepare_triangular_contact_schur(problem["sparse_c"], problem["operator"],
        problem["drive"], **arguments)


def solve(plan, problem, step, **changes):
    arguments = dict(step_seconds=step, old=problem["old"],
        stage_currents=problem["currents"], stage_drives=problem["stage_drives"],
        passive_product=lambda value: problem["operator"] @ value, check=lambda: None)
    arguments.update(changes)
    return plan.solve_stages(**arguments)


class TriangularContactSchurTests(unittest.TestCase):
    def test_full_two_stage_dense_parity_including_contact_charge_noise_and_clipping(self):
        for clipped in (False, True):
            with self.subTest(clipped=clipped):
                problem, step = finite_problem(clipped=clipped), .035
                info = {}
                plan = prepare(problem, fallback_info=info)
                self.assertIsNotNone(plan)
                self.assertTrue(info["supported"])
                self.assertEqual(plan.state_count, SIZE)
                self.assertLessEqual(plan.estimated_workspace_bytes, 32 * 1024**2)
                old, currents = problem["old"].copy(), problem["currents"].copy()
                counted = []

                def passive(value):
                    counted.append(value.copy())
                    return problem["operator"] @ value

                actual = solve(plan, problem, step, passive_product=passive)
                expected = dense_stage_oracle(problem, step, clipped=clipped)
                self.assertEqual(actual.shape, (2, SIZE))
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
                self.assertFalse(np.shares_memory(actual, problem["old"]))
                self.assertFalse(np.shares_memory(actual, problem["currents"]))
                np.testing.assert_array_equal(problem["old"], old)
                np.testing.assert_array_equal(problem["currents"], currents)
                self.assertEqual(len(counted), 1)
                np.testing.assert_array_equal(counted[0], old)
                self.assertGreater(np.max(np.abs(actual[:, CONTACTS] - old[CONTACTS])), 1e-5)
                residual, charge, right = full_residual(problem, step, actual)
                scale = 1e-11 + 1e-8 * np.maximum(np.abs(charge), np.abs(right))
                self.assertLessEqual(float(np.max(np.abs(residual) / scale)), 1.)
                correction = np.linalg.solve(problem["c"], residual.T).T
                self.assertLessEqual(float(np.max(np.abs(correction) /
                    (1e-10 + 1e-8 * np.maximum(np.abs(old), np.abs(actual))))), 1.)
                # Contact recovery must preserve stage work as well as state.
                for index in range(2):
                    first = problem["stage_drives"][index].evaluate(actual[index])
                    second = problem["stage_drives"][index].evaluate(expected[index])
                    for field in ("controlled_voltage_source_power_watt",
                                  "controlled_branch_dissipation_watt",
                                  "controlled_nodal_current_power_watt", "buffer_shunt_power_watt"):
                        self.assertAlmostEqual(getattr(first, field), getattr(second, field), delta=2e-11)
                defect = 3. * (actual[0] - old) - (actual[1] - old)
                expected_defect = 3. * (expected[0] - old) - (expected[1] - old)
                self.assertAlmostEqual(.125 * (defect @ problem["c"] @ defect),
                    .125 * (expected_defect @ problem["c"] @ expected_defect), delta=2e-12)
                if not clipped:
                    first_metrics = plan.metadata
                    self.assertEqual(first_metrics["factor_builds"], 1)
                    self.assertEqual(first_metrics["stage_solves"], 1)
                    repeated = solve(plan, problem, step)
                    np.testing.assert_array_equal(repeated, actual)
                    self.assertEqual(plan.metadata["factor_builds"], 1)
                    adjacent = np.nextafter(step, np.inf)
                    solve(plan, problem, adjacent)
                    self.assertEqual(plan.metadata["factor_builds"], 2)
                    self.assertEqual(plan.metadata["cached_step_seconds"], adjacent)
                    # Returning to h rebuilds it: there is only one exact-h
                    # entry, and neighboring floats are never rounded together.
                    solve(plan, problem, step)
                    metrics = plan.metadata
                    self.assertEqual(metrics["factor_builds"], 3)
                    self.assertEqual(metrics["stage_solves"], 4)
                    self.assertEqual(metrics["band_factorizations"], 3)
                    self.assertEqual(metrics["downstream_factorizations"], 3)
                    self.assertEqual(metrics["contact_factorizations"],
                        3 * metrics["distinct_contact_blocks"])
                    self.assertEqual(metrics["cached_step_seconds"], step)

    def test_cancellation_heavy_common_mode_preserves_original_full_equation(self):
        problem, step = finite_problem(cancellation=True), .02
        problem["old"] = np.full(SIZE, .3)
        # Prescribe nonzero common-mode changes plus much smaller contact
        # differences, then manufacture currents from the ORIGINAL equations.
        # The capacitance ratio exceeds 10^6; R*sum(g) exceeds 10^8.
        expected = np.vstack((problem["old"] + .00015, problem["old"] + .0002))
        expected[0, CONTACTS] += 1e-8 * np.sin(np.arange(32))
        expected[1, CONTACTS] += 2e-8 * np.cos(np.arange(32))
        charge = (problem["c"] @ (expected - problem["old"]).T).T
        forces = np.linalg.solve(A, charge) / step
        for index in range(2):
            problem["currents"][index] = forces[index] + problem["operator"] @ expected[index]
            problem["currents"][index, BUFFERS] -= problem["stage_drives"][index].evaluate(
                expected[index]).current_amp
        plan = prepare(problem)
        self.assertIsNotNone(plan)
        actual = solve(plan, problem, step)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-8)
        residual, charge, right = full_residual(problem, step, actual)
        scale = 2e-10 + 2e-7 * np.maximum(np.abs(charge), np.abs(right))
        self.assertLessEqual(float(np.max(np.abs(residual) / scale)), 1.)
        correction = np.linalg.solve(problem["c"], residual.T).T
        self.assertLessEqual(float(np.max(np.abs(correction))), 2e-7)

    def test_actual_topology_workspace_and_callback_refusal_leave_inputs_intact(self):
        problem = finite_problem()
        with self.assertRaises(SolverConvergenceError):
            prepare(problem, max_workspace_bytes=1)
        cases = []
        for matrix_name, first, second in (("c", 2, 98), ("base", 2, 98), ("c", 66, 82)):
            changed = dict(problem)
            matrix = problem[matrix_name].copy()
            _edge(matrix, first, second, .001)
            if matrix_name == "c":
                changed["sparse_c"] = csr_matrix(matrix)
            else:
                changed["operator"] = _ConductanceWithStars(csr_matrix(matrix), problem["stars"])
            cases.append((f"{matrix_name}:{first}:{second}", changed))
        extra_contact_shunt = problem["base"].copy()
        extra_contact_shunt[66, 66] += .001
        changed = dict(problem, operator=_ConductanceWithStars(csr_matrix(extra_contact_shunt), problem["stars"]))
        cases.append(("unhandled contact shunt", changed))
        for name, changed in cases:
            with self.subTest(name=name):
                info = {}
                self.assertIsNone(prepare(changed, fallback_info=info))
                self.assertFalse(info["supported"])
                self.assertTrue(info["reason"])
        stopped = RuntimeError("owned deadline")

        def stop():
            raise stopped

        with self.assertRaises(RuntimeError) as result:
            prepare(problem, check=stop)
        self.assertIs(result.exception, stopped)
        plan = prepare(problem)
        old, currents = problem["old"].copy(), problem["currents"].copy()
        with self.assertRaises(RuntimeError) as result:
            solve(plan, problem, .03, check=stop)
        self.assertIs(result.exception, stopped)
        np.testing.assert_array_equal(problem["old"], old)
        np.testing.assert_array_equal(problem["currents"], currents)


class TriangularContactSchurIntegrationTests(unittest.TestCase):
    """Requires the UNAPPLIED integration patch; do not silently skip it."""

    def test_original_radau_gates_work_stage_order_counters_and_transactional_commit(self):
        problem, duration = finite_problem(clipped=True), .035
        outputs = []
        work_fields = ("controlled_voltage_source_work_joule",
            "controlled_branch_dissipation_joule", "controlled_nodal_current_work_joule",
            "passive_norton_dissipation_joule", "controlled_buffer_shunt_quadrature_joule",
            "controlled_port_corrected_norton_dissipation_joule", "time_varying_current_work_joule")

        def fixture(strategy):
            network = ConditionedRCNetwork(state_ids=tuple(f"finite:{i}" for i in range(SIZE)),
                initial_volts=problem["initial"], capacitance_to_reference_farad=problem["ground"],
                mutual_capacitance_edges=problem["cap_edges"])
            evaluated = []

            class Forcing:
                def _at(self, time, kind):
                    if not 0. <= time <= duration:
                        raise AssertionError("forcing escaped the original physical interval")
                    evaluated.append((kind, time))
                    noise = .003 * np.sin(np.arange(64))
                    noise[0] = .5 - time / duration
                    current = (.004 + .003 * time / duration) * np.sin(np.arange(SIZE))
                    return current, problem["drive"].with_input_noise_volts(noise)

                def evaluate(self, time):
                    return self._at(time, "evaluate")

                def evaluate_left(self, time):
                    return self._at(time, "evaluate_left")

                def work_rates(self, time, state):
                    # This finite fixture supplies nodal current injections;
                    # it has no separately mapped physical voltage ports.
                    return np.zeros(4)

                def work_rates_left(self, time, state):
                    return np.zeros(4)

            arguments = dict(duration_seconds=duration,
                conductance_to_reference_siemens=problem["shunts"], resistor_edges=problem["edges"],
                drive_current_amp=problem["currents"][0],
                observation_matrix=csr_matrix(np.eye(SIZE)), feedthrough_volts=np.zeros(SIZE),
                injected_charge_coulomb=problem["injected"], controlled_drives=problem["drive"],
                passive_stars=problem["stars"], time_varying_forcing=Forcing(), check=lambda: None,
                controls=controls(integration_method="radau_iia_2", radau_preconditioner="contact_block",
                    radau_linear_solver=strategy, max_refinements=12,
                    max_workspace_bytes=64 * 1024**2))
            return network, arguments, evaluated

        for strategy in ("gmres", "triangular_contact_schur"):
            network, arguments, evaluated = fixture(strategy)
            before = network.volts
            if strategy == "gmres":
                result = network.advance(**arguments, commit=False)
            else:
                refused = AssertionError("the direct path called an iterative solver/preconditioner")
                with patch("conditioned_radau_step.gmres", side_effect=refused), \
                     patch("conditioned_radau_step.LinearOperator", side_effect=refused), \
                     patch("conditioned_array_solver.prepare_contact_blocks", side_effect=refused):
                    result = network.advance(**arguments, commit=False)
                self.assertEqual(result.newton_iterations, 0)
                self.assertEqual(result.linear_iterations, 0)
                self.assertEqual(result.radau_preconditioner, "not_used")
            self.assertEqual(result.radau_linear_solver, strategy)
            self.assertEqual(network.volts, before)
            self.assertEqual(len(result.observation.end_volts), SIZE)
            self.assertEqual(len(evaluated) % 2, 0)
            self.assertGreater(len(evaluated), 0)
            for first, second in zip(evaluated[::2], evaluated[1::2]):
                self.assertEqual((first[0], second[0]), ("evaluate", "evaluate_left"))
                self.assertLess(first[1], second[1])
            self.assertEqual(evaluated[-1], ("evaluate_left", duration))
            self.assertLessEqual(result.maximum_scaled_midpoint_residual, 1.)
            self.assertLessEqual(result.maximum_scaled_partition_error, 1.)
            self.assertGreaterEqual(result.subintervals, 2)
            self.assertGreater(result.operator_products, 0)
            scale = arguments["controls"].energy_atol_joule + arguments["controls"].relative_tolerance * max(
                abs(result.observation.start_energy_joule), abs(result.observation.end_energy_joule),
                abs(duration * (problem["currents"][0] @ np.asarray(result.observation.mean_volts))),
                abs(result.controlled_nodal_current_work_joule),
                abs(result.passive_norton_dissipation_joule), abs(result.time_varying_current_work_joule))
            self.assertLessEqual(abs(result.energy_balance_residual_joule), scale)
            self.assertLessEqual(result.numerical_dissipation_joule, scale)
            self.assertLessEqual(abs(result.energy_balance_residual_joule + result.numerical_dissipation_joule), scale)
            network.commit_preview(result)
            self.assertEqual(network.volts, result.observation.end_volts)
            with self.assertRaises(ValueError):
                network.commit_preview(result)
            outputs.append(result)
        for field in ("end_volts", "mean_volts"):
            np.testing.assert_allclose(getattr(outputs[0].observation, field),
                getattr(outputs[1].observation, field), rtol=2e-5, atol=4e-8)
        np.testing.assert_allclose([getattr(outputs[0], name) for name in work_fields],
            [getattr(outputs[1], name) for name in work_fields], rtol=2e-5, atol=4e-8)
        network, arguments, _ = fixture("triangular_contact_schur")
        before = network.volts
        arguments["controls"] = replace(arguments["controls"], max_operator_products=1)
        with self.assertRaises(SolverConvergenceError):
            network.advance(**arguments)
        self.assertEqual(network.volts, before)
        plan = prepare(problem)
        original_solve = type(plan).solve_stages

        def corrupt_contact(self, **kwargs):
            value = original_solve(self, **kwargs).copy()
            value[0, 66] += .001
            return value

        network, arguments, _ = fixture("triangular_contact_schur")
        before = network.volts
        with patch.object(type(plan), "solve_stages", corrupt_contact):
            with self.assertRaises(SolverConvergenceError):
                network.advance(**arguments)
        self.assertEqual(network.volts, before)


if __name__ == "__main__":
    unittest.main()
