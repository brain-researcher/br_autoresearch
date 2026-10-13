"""Independent algebra fixtures; no device constants or production run."""
from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from analog_conditioning import ControlledVoltageDrives, SourceConditioning
from conditioned_array_solver import ConditionedRCNetwork, ConditionedSolverControls
from sparse_array_rc import PassiveStarConductance, SolverConvergenceError


def controls(**changes):
    values = dict(relative_tolerance=2e-5, voltage_atol=2e-8, charge_atol_coulomb=1e-11,
        max_refinements=8, max_operator_products=200000,
        linear_relative_tolerance=1e-12, linear_charge_atol_coulomb=1e-14,
        energy_atol_joule=2e-8, max_newton_iterations=8, max_linear_iterations=20000,
        gmres_restart=12, max_workspace_bytes=4 * 1024**2)
    values.update(changes)
    return ConditionedSolverControls(**values)


def channel(**changes):
    values = dict(gain=1.2, offset_volts=.05, drive_lower_volts=-10., drive_upper_volts=10.,
        input_capacitance_farad=0., input_conductance_siemens=0., buffer_capacitance_farad=1.,
        buffer_conductance_siemens=2., output_conductance_siemens=.5,
        filter_capacitance_farad=3., filter_leak_conductance_siemens=.2,
        initial_buffer_volts=-.1, initial_filter_volts=.05)
    values.update(changes)
    return SourceConditioning(**values)


def fixture(*, clipped=False):
    size = 192
    selected = channel(gain=1., offset_volts=0., drive_lower_volts=-.5, drive_upper_volts=.5) if clipped else channel()
    drives = ControlledVoltageDrives(channels=(selected,) * 64, source_nodes=tuple(range(64)),
                                     buffer_nodes=tuple(range(64, 128)), state_count=size)
    initial3 = np.array([2., 1., .4] if clipped else [.1, -.1, .05])
    initial = np.repeat(initial3, 64)
    capacitance_edges = [] if clipped else [(i, i + 64, .2) for i in range(64)]
    network = ConditionedRCNetwork(state_ids=tuple(f"v:{i}" for i in range(size)), initial_volts=initial,
        capacitance_to_reference_farad=np.repeat([2., 1., 3.], 64), mutual_capacitance_edges=capacitance_edges)
    base3 = np.diag([0. if clipped else .4, 2., .2])
    edge = np.array([0., 1., -1.])
    g3 = base3 + .5 * np.outer(edge, edge)
    c3 = np.diag([2., 1., 3.])
    i3 = np.zeros(3) if clipped else np.array([.3, 0., 0.])
    star = None
    if not clipped:
        cedge = np.array([1., -1., 0.])
        c3 += .2 * np.outer(cedge, cedge)
        h = np.array([.15, 0., .1])
        g3 += np.diag(h) - .4 * np.outer(h, h) / (1 + .4 * h.sum())
        i3 += h * .2 / (1 + .4 * h.sum())
        star = PassiveStarConductance(state_count=size,
            groups=tuple(((i, i + 128), (.15, .1), .4) for i in range(64)))
    current = np.repeat(i3, 64)
    args = dict(duration_seconds=.2, conductance_to_reference_siemens=np.repeat(np.diag(base3), 64),
        resistor_edges=[(i + 64, i + 128, .5) for i in range(64)], drive_current_amp=current,
        observation_matrix=csr_matrix((np.ones(3), (range(3), (0, 64, 128))), shape=(3, size)),
        feedthrough_volts=np.zeros(3), injected_charge_coulomb=np.zeros(size), controls=controls(),
        controlled_drives=drives, passive_stars=star, check=lambda: None)
    return network, args, c3, g3, i3, initial3, selected


def exact_linear(c3, g3, i3, initial, selected, duration, *, fixed_target=None):
    # Independent continuous-time affine matrix exponential, not midpoint replay.
    active = np.zeros((3, 3))
    offset = np.zeros(3)
    if fixed_target is None:
        active[1, 0] = selected.buffer_conductance_siemens * selected.gain
        offset[1] = selected.buffer_conductance_siemens * selected.offset_volts
    else:
        offset[1] = selected.buffer_conductance_siemens * fixed_target
    a4 = np.zeros((4, 4))
    a4[:3, :3] = np.linalg.solve(c3, -g3 + active)
    a4[:3, 3] = np.linalg.solve(c3, i3 + offset)
    extended = np.zeros((8, 8))
    extended[:4, :4] = a4
    extended[4:, :4] = np.eye(4)
    result = expm(extended * duration) @ np.r_[initial, 1., np.zeros(4)]
    return result[:3], result[4:7] / duration, a4


class ConditionedArraySolverTests(unittest.TestCase):
    def test_continuous_cubic_forcing_and_four_physical_works_match_exponential(self):
        network, args, c3, g3, i3, initial, selected = fixture()
        eta = np.array([.10, .30, -.20, .15])
        access = np.array([.05, -.10, .30, -.20])
        amplifier = np.array([.04, .15, -.10, .08])
        remote, field, branch_g = .4, .1, .05
        # Reserve .05S of the existing .2S filter shunt for a prescribed
        # voltage branch. Its held field voltage contributes this base drive;
        # the continuous series-voltage noise contributes g*e(t) separately.
        i3 = i3 + np.array([0., 0., branch_g * field])
        args["drive_current_amp"] = np.repeat(i3, 64)
        base = args["controlled_drives"]
        evaluated = []

        class Forcing:
            def evaluate(self, local_time):
                evaluated.append(local_time)
                basis = np.array([1., local_time, local_time**2, local_time**3])
                current = np.repeat([eta @ basis, 0., branch_g * (access @ basis)], 64)
                return current, base.with_input_noise_volts(np.full(64, amplifier @ basis))

            def work_rates(self, local_time, middle):
                basis = np.array([1., local_time, local_time**2, local_time**3])
                current, voltage = eta @ basis, access @ basis
                return (current * float(np.sum(middle[:64] - remote)),
                    branch_g * voltage * float(np.sum(field + voltage - middle[128:])),
                    64 * field * branch_g * voltage, 64 * remote * current)

        # Autonomous continuous-time system on [v_source,v_buffer,v_filter,
        # 1,t,t^2,t^3]. Neither its evolution nor moment integrals use midpoint
        # partitions, coefficient samples or numerical solver results.
        active = np.zeros((3, 3))
        active[1, 0] = selected.buffer_conductance_siemens * selected.gain
        generator = np.zeros((7, 7))
        generator[:3, :3] = np.linalg.solve(c3, -g3 + active)
        coefficients = np.vstack((eta, selected.buffer_conductance_siemens * selected.gain * amplifier,
                                  branch_g * access))
        coefficients[:, 0] += i3 + np.array([0., selected.buffer_conductance_siemens * selected.offset_volts, 0.])
        generator[:3, 3:] = np.linalg.solve(c3, coefficients)
        generator[4, 3], generator[5, 4], generator[6, 5] = 1., 2., 3.
        z = np.r_[initial, 1., 0., 0., 0.]
        augmented = np.zeros((14, 14))
        augmented[:7, :7], augmented[7:, :7] = generator, np.eye(7)
        exact = expm(augmented * .2) @ np.r_[z, np.zeros(7)]
        moment_generator = np.kron(generator, np.eye(7)) + np.kron(np.eye(7), generator)
        moment_augmented = np.zeros((98, 98))
        moment_augmented[:49, :49], moment_augmented[49:, :49] = moment_generator, np.eye(49)
        integral = (expm(moment_augmented * .2) @ np.r_[np.outer(z, z).ravel(), np.zeros(49)])[49:].reshape(7, 7)
        one, source, buffer, filtered = np.eye(7)[[3, 0, 1, 2]]
        eta_vector, access_vector, amp_vector = (np.r_[np.zeros(3), item] for item in (eta, access, amplifier))
        expected_works = 64 * np.array([
            eta_vector @ integral @ (source - remote * one),
            branch_g * access_vector @ integral @ (field * one + access_vector - filtered),
            field * branch_g * access_vector @ integral @ one,
            remote * eta_vector @ integral @ one])
        expected_nodal = 64 * (eta_vector @ integral @ source
                              + branch_g * access_vector @ integral @ filtered)
        result = network.advance(**args, time_varying_forcing=Forcing(), commit=False)
        np.testing.assert_allclose(result.observation.end_volts, np.repeat(exact[:3], 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.observation.mean_volts, np.repeat(exact[7:10] / .2, 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.physical_noise_work_joule, expected_works, rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.time_varying_current_work_joule, expected_nodal, rtol=1e-4, atol=2e-7)
        target = selected.gain * (source + amp_vector) + selected.offset_volts * one
        expected_controlled = 64 * selected.buffer_conductance_siemens * target @ integral @ (target - buffer)
        np.testing.assert_allclose(result.controlled_voltage_source_work_joule, expected_controlled, rtol=1e-4, atol=2e-7)
        self.assertEqual(len(evaluated), 2 * result.subintervals - 1)
        self.assertTrue(all(0. < point < .2 for point in evaluated))
        self.assertGreater(len(set(evaluated)), 3)
        mean_basis = np.array([1., .2 / 2, .2**2 / 3, .2**3 / 4])
        false_mean_product = 64 * ((eta @ mean_basis) * exact[7]
                                  + branch_g * (access @ mean_basis) * exact[9])
        self.assertGreater(abs(expected_nodal - false_mean_product), 1e-7)
        base_work = .2 * float(args["drive_current_amp"] @ np.asarray(result.observation.mean_volts))
        self.assertAlmostEqual(result.observation.input_work_joule,
            base_work + result.controlled_nodal_current_work_joule + result.time_varying_current_work_joule, places=11)
        self.assertLess(abs(result.energy_balance_residual_joule), 1e-9)
        self.assertEqual(network.volts, tuple(np.repeat(initial, 64)))
        network.commit_preview(result)
        self.assertEqual(network.volts, result.observation.end_volts)

    def test_time_varying_amplifier_noise_crosses_clipping_before_state_propagation(self):
        network, args, c3, g3, i3, initial, selected = fixture(clipped=True)
        # Job47123089: the controlled-source work error at64/128/256 parts
        # was67.293/16.817/4.204, while the256-part endpoint/mean errors were
        # .0234/.0351. Allow two further midpoint refinements of this steep
        # clip-crossing fixture; retain its tolerances and independent oracle.
        # The production recipe's selected eight refinements remain unchanged.
        args["controls"] = controls(max_refinements=10)
        base = args["controlled_drives"]

        class Forcing:
            def evaluate(self, local_time):
                return np.zeros(192), base.with_input_noise_volts(np.full(64, -20 * local_time))

            def work_rates(self, local_time, middle):
                return (0., 0., 0., 0.)

        # Source voltage stays exactly2V. The requested target2−20t crosses
        # +.5 at.075s and−.5 at.125s. Integrate those three affine pieces
        # independently, carrying every capacitor state and its aperture sum.
        voltage, integrated = initial.copy(), np.zeros(3)
        for start, end, intercept, slope in ((0., .075, .5, 0.), (.075, .125, 2., -20.),
                                            (.125, .2, -.5, 0.)):
            generator = np.zeros((5, 5))
            generator[:3, :3] = np.linalg.solve(c3, -g3)
            generator[:3, 3] = np.linalg.solve(c3, i3 + [0., selected.buffer_conductance_siemens * intercept, 0.])
            generator[:3, 4] = np.linalg.solve(c3, [0., selected.buffer_conductance_siemens * slope, 0.])
            generator[4, 3] = 1.
            augmented = np.zeros((10, 10))
            augmented[:5, :5], augmented[5:, :5] = generator, np.eye(5)
            exact = expm(augmented * (end - start)) @ np.r_[voltage, 1., start, np.zeros(5)]
            voltage = exact[:3]
            integrated += exact[5:8]
        result = network.advance(**args, time_varying_forcing=Forcing())
        np.testing.assert_allclose(result.observation.end_volts, np.repeat(voltage, 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.observation.mean_volts, np.repeat(integrated / .2, 64), rtol=1e-4, atol=2e-7)
        self.assertEqual(result.time_varying_current_work_joule, 0.)
        self.assertEqual(result.physical_noise_work_joule, (0., 0., 0., 0.))
        np.testing.assert_array_equal(base.evaluate(np.repeat(initial, 64)).current_amp, np.ones(64))

    def test_every_physical_work_channel_controls_refinement_and_failure_is_atomic(self):
        for channel_index in range(4):
            _, args, *_ = fixture(clipped=True)
            equilibrium = np.repeat([2., 7 / 15, 1 / 3], 64)
            network = ConditionedRCNetwork(state_ids=tuple(f"v:{i}" for i in range(192)),
                initial_volts=equilibrium, capacitance_to_reference_farad=np.repeat([2., 1., 3.], 64),
                mutual_capacitance_edges=())
            args["controls"] = controls(max_refinements=1, relative_tolerance=0., voltage_atol=1e-6,
                                        energy_atol_joule=1e-12)
            base = args["controlled_drives"]

            class QuadratureOnly:
                def evaluate(self, local_time):
                    return np.zeros(192), base

                def work_rates(self, local_time, middle):
                    values = np.zeros(4)
                    values[channel_index] = local_time**4
                    return values

            with self.subTest(physical_channel=channel_index), self.assertRaisesRegex(SolverConvergenceError, "partition agreement"):
                network.advance(**args, time_varying_forcing=QuadratureOnly())
            self.assertEqual(network.volts, tuple(equilibrium))
        for bad_part in ("current", "work"):
            network, args, *_ = fixture()
            before = network.volts
            args["injected_charge_coulomb"] = np.ones(192) * .01
            base = args["controlled_drives"]

            class InvalidLate:
                def evaluate(self, local_time):
                    vector = np.zeros(192)
                    if bad_part == "current" and local_time > .1:
                        vector[0] = np.nan
                    return vector, base

                def work_rates(self, local_time, middle):
                    return (np.inf if bad_part == "work" and local_time > .1 else 0., 0., 0., 0.)

            with self.subTest(bad_part=bad_part), self.assertRaises(ValueError):
                network.advance(**args, time_varying_forcing=InvalidLate())
            self.assertEqual(network.volts, before)
        first, first_args, *_ = fixture()
        second, second_args, *_ = fixture()
        legacy = first.advance(**first_args)
        explicit = second.advance(**second_args, time_varying_forcing=None)
        self.assertEqual(legacy, explicit)
        self.assertEqual(explicit.time_varying_current_work_joule, 0.)
        self.assertEqual(explicit.physical_noise_work_joule, (0., 0., 0., 0.))

    def test_joint_linear_endpoint_and_aperture_mean_match_independent_exponential(self):
        network, args, c3, g3, i3, initial, selected = fixture()
        expected_end, expected_mean, _ = exact_linear(c3, g3, i3, initial, selected, .2)
        result = network.advance(**args, commit=False)
        self.assertEqual(network.volts, tuple(np.repeat(initial, 64)))
        np.testing.assert_allclose(result.observation.end_volts, np.repeat(expected_end, 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.observation.mean_volts, np.repeat(expected_mean, 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.observation.output_mean_volts, expected_mean, rtol=1e-4, atol=2e-7)
        self.assertGreater(result.newton_iterations, 0)
        self.assertGreater(result.linear_iterations, 0)
        self.assertGreater(result.operator_products, 0)
        network.commit_preview(result)
        self.assertEqual(network.volts, result.observation.end_volts)
        with self.assertRaises(ValueError):
            network.commit_preview(result)

    def test_clip_target_and_recovery_keep_loaded_state_and_once_only_charge(self):
        network, args, c3, g3, i3, initial, selected = fixture(clipped=True)
        expected, mean, _ = exact_linear(c3, g3, i3, initial, selected, .2, fixed_target=.5)
        first = network.advance(**args)
        np.testing.assert_allclose(first.observation.end_volts, np.repeat(expected, 64), rtol=1e-4, atol=2e-7)
        self.assertGreater(first.observation.end_volts[64], .5)  # No false hard output clamp.
        before = np.array(network.volts)
        charge = np.zeros(192)
        charge[:64] = -8.  # C_source=2 and source goes from+2 to−2.
        args["injected_charge_coulomb"] = charge
        after_boundary = before[[0, 64, 128]].copy()
        after_boundary[0] = -2.
        expected2, _, _ = exact_linear(c3, g3, i3, after_boundary, selected, .2, fixed_target=-.5)
        second = network.advance(**args, commit=False)
        np.testing.assert_allclose(second.observation.start_volts[64:], before[64:], atol=1e-14)
        np.testing.assert_allclose(second.observation.end_volts, np.repeat(expected2, 64), rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(second.boundary.transition.injected_charge_coulomb, charge)
        self.assertEqual(network.volts, tuple(before))

    def test_terminal_work_and_branch_heat_match_independent_quadratic_integrals(self):
        network, args, c3, g3, i3, initial, selected = fixture()
        result = network.advance(**args)
        _, _, a4 = exact_linear(c3, g3, i3, initial, selected, .2)
        # vec(z z') follows a 16-dimensional linear ODE. Integrate it exactly.
        quadratic = np.kron(a4, np.eye(4)) + np.kron(np.eye(4), a4)
        augmented = np.zeros((32, 32))
        augmented[:16, :16] = quadratic
        augmented[16:, :16] = np.eye(16)
        z = np.r_[initial, 1.]
        integrated = (expm(augmented * .2) @ np.r_[np.outer(z, z).ravel(), np.zeros(16)])[16:].reshape(4, 4)
        target = np.array([selected.gain, 0., 0., selected.offset_volts])
        buffer = np.array([0., 1., 0., 0.])
        branch = target - buffer
        gb = selected.buffer_conductance_siemens
        expected_source = 64 * gb * float(target @ integrated @ branch)
        expected_heat = 64 * gb * float(branch @ integrated @ branch)
        np.testing.assert_allclose(result.controlled_voltage_source_work_joule, expected_source, rtol=1e-4, atol=2e-7)
        np.testing.assert_allclose(result.controlled_branch_dissipation_joule, expected_heat, rtol=1e-4, atol=2e-7)
        self.assertAlmostEqual(result.controlled_voltage_source_work_joule - result.controlled_branch_dissipation_joule,
            result.controlled_nodal_current_work_joule - result.controlled_buffer_shunt_quadrature_joule, places=11)
        base_work = .2 * float(np.repeat(i3, 64) @ np.array(result.observation.mean_volts))
        delta_energy = result.observation.end_energy_joule - result.observation.start_energy_joule
        self.assertAlmostEqual(delta_energy, base_work + result.controlled_voltage_source_work_joule
                               - result.controlled_port_corrected_norton_dissipation_joule, places=9)
        self.assertLess(abs(result.energy_balance_residual_joule), 1e-9)

    def test_all_numeric_resource_refusals_preserve_full_state_and_unapplied_charge(self):
        network, args, *_ = fixture()
        before = network.volts
        args["conductance_to_reference_siemens"][64] = 0.
        with self.assertRaisesRegex(ValueError, "controlled branch conductance"):
            network.advance(**args, commit=False)
        self.assertEqual(network.volts, before)
        for limits in (dict(max_operator_products=1), dict(max_linear_iterations=1),
                       dict(max_newton_iterations=1), dict(max_workspace_bytes=1),
                       dict(max_refinements=1, relative_tolerance=0., voltage_atol=1e-14, energy_atol_joule=1e-14)):
            network, args, *_ = fixture()
            before = network.volts
            args["controls"] = controls(**limits)
            args["injected_charge_coulomb"] = np.ones(192) * .01
            with self.subTest(limits=limits), self.assertRaises(SolverConvergenceError):
                network.advance(**args, commit=False)
            self.assertEqual(network.volts, before)
        network, args, *_ = fixture()
        pending = network.advance(**args, commit=False)
        before = network.volts
        args["controls"] = controls(max_operator_products=1)
        with self.assertRaises(SolverConvergenceError):
            network.advance(**args, commit=False)
        self.assertEqual(network.volts, before)
        with self.assertRaises(ValueError):
            network.commit_preview(pending)
        args["check"] = lambda: (_ for _ in ()).throw(TimeoutError("fixture deadline"))
        with self.assertRaises(TimeoutError):
            network.advance(**args)
        self.assertEqual(network.volts, before)


if __name__ == "__main__":
    unittest.main()
