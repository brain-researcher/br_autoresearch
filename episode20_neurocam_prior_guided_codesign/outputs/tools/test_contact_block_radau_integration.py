"""Contact-block GMRES checks against an independent continuous-time solution.

The small algebra fixture retains all 64 controlled drives and every declared
contact state. It is not a device realization, saved world or parameter sweep.
"""
from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix

from analog_conditioning import ControlledVoltageDrives
from conditioned_array_solver import ConditionedRCNetwork
from sparse_array_rc import PassiveStarConductance, SolverConvergenceError
from test_conditioned_array_solver import channel, controls, fixture as generic_fixture


SELECTED = np.r_[0, 64, np.arange(128, 145)]
WORK_FIELDS = (
    "controlled_voltage_source_work_joule", "controlled_branch_dissipation_joule",
    "controlled_nodal_current_work_joule", "passive_norton_dissipation_joule",
    "controlled_buffer_shunt_quadrature_joule",
    "controlled_port_corrected_norton_dissipation_joule",
)


def contact_problem(strategy):
    size, duration = 145, .04
    selected = channel(gain=1.2, offset_volts=0., drive_lower_volts=-.5,
        drive_upper_volts=.5, buffer_conductance_siemens=2.)
    drives = ControlledVoltageDrives(channels=(selected,) * 64,
        source_nodes=tuple(range(64)), buffer_nodes=tuple(range(64, 128)), state_count=size)
    ground = np.r_[np.ones(64), np.full(64, .7), .9, np.zeros(16)]
    initial = np.zeros(size)
    initial[[0, 64, 128]] = (.1, .03, .02)
    branch_cap = np.linspace(.04, .2, 16)
    cap_edges = [(128, 129 + index, float(value)) for index, value in enumerate(branch_cap)]
    # Couplings beyond the contact block must remain in the actual residual,
    # including their diagonal contributions in the principal preconditioner.
    cap_edges.extend(((0, 128, .06), (0, 64, .04)))
    network = ConditionedRCNetwork(state_ids=tuple(f"fixture:{i}" for i in range(size)),
        initial_volts=initial, capacitance_to_reference_farad=ground,
        mutual_capacitance_edges=cap_edges)
    shunts = np.r_[np.full(64, .2), np.full(64, 2.), .03, np.zeros(16)]
    star_weights = np.r_[.02, np.linspace(.05, .8, 16)]
    resistance = .3
    stars = PassiveStarConductance(state_count=size,
        groups=((tuple(range(128, 145)), tuple(star_weights), resistance),))
    current = np.zeros(size)
    current[128:] = star_weights * .02 / (1 + resistance * star_weights.sum())
    selected_controls = controls(integration_method="radau_iia_2",
        radau_preconditioner=strategy, relative_tolerance=1e-7,
        voltage_atol=1e-10, charge_atol_coulomb=1e-12,
        energy_atol_joule=1e-10, max_workspace_bytes=16 * 1024**2)
    arguments = dict(duration_seconds=duration, conductance_to_reference_siemens=shunts,
        resistor_edges=((0, 128, .15),), drive_current_amp=current,
        observation_matrix=csr_matrix((np.ones(len(SELECTED)),
            (np.arange(len(SELECTED)), SELECTED)), shape=(len(SELECTED), size)),
        feedthrough_volts=np.zeros(len(SELECTED)), injected_charge_coulomb=np.zeros(size),
        controls=selected_controls, controlled_drives=drives, passive_stars=stars,
        check=lambda: None)

    # An independent dense, constant affine generator on the only nonzero
    # 19-state component. Its solution does not use Radau stages, partitions,
    # passive-operator products or a preconditioner implementation.
    c = np.diag(ground[SELECTED])
    g = np.diag(shunts[SELECTED])
    local = {int(node): index for index, node in enumerate(SELECTED)}
    for first, second, value in cap_edges:
        edge = np.zeros(len(SELECTED))
        edge[local[first]], edge[local[second]] = 1., -1.
        c += value * np.outer(edge, edge)
    edge = np.zeros(len(SELECTED))
    edge[0], edge[2] = 1., -1.
    g += .15 * np.outer(edge, edge)
    g[2:, 2:] += (np.diag(star_weights)
        - resistance * np.outer(star_weights, star_weights) / (1 + resistance * star_weights.sum()))
    active = np.zeros_like(g)
    active[1, 0] = selected.buffer_conductance_siemens * selected.gain
    count = len(SELECTED)
    affine = np.zeros((count + 1, count + 1))
    affine[:count, :count] = np.linalg.solve(c, active - g)
    affine[:count, count] = np.linalg.solve(c, current[SELECTED])
    augmented = np.zeros((2 * count + 1, 2 * count + 1))
    augmented[:count + 1, :count + 1] = affine
    augmented[count + 1:, :count] = np.eye(count)
    exact = expm(duration * augmented) @ np.r_[initial[SELECTED], 1., np.zeros(count)]
    return network, arguments, exact[:count], exact[count + 1:] / duration


class ContactBlockRadauIntegrationTests(unittest.TestCase):
    def test_exact_equations_state_work_and_preview_commit_remain_unchanged(self):
        results = []
        for strategy in ("stage_jacobi", "contact_block"):
            network, arguments, endpoint, mean = contact_problem(strategy)
            before = network.volts
            result = network.advance(**arguments, commit=False)
            self.assertEqual(network.volts, before)
            self.assertEqual(result.radau_preconditioner, strategy)
            self.assertEqual(len(result.observation.end_volts), 145)
            np.testing.assert_allclose(np.asarray(result.observation.end_volts)[SELECTED],
                endpoint, rtol=1e-6, atol=2e-9)
            np.testing.assert_allclose(np.asarray(result.observation.mean_volts)[SELECTED],
                mean, rtol=1e-6, atol=2e-9)
            self.assertLessEqual(result.maximum_scaled_midpoint_residual, 1.)
            self.assertLessEqual(result.maximum_scaled_partition_error, 1.)
            # Unchanged physical and numerical energy acceptance is still
            # applied after the full coupled solve.
            scale = arguments["controls"].energy_atol_joule + arguments["controls"].relative_tolerance * max(
                abs(result.observation.start_energy_joule), abs(result.observation.end_energy_joule),
                abs(result.observation.input_work_joule), abs(result.passive_norton_dissipation_joule),
                abs(result.controlled_nodal_current_work_joule))
            self.assertLessEqual(abs(result.energy_balance_residual_joule), scale)
            self.assertLessEqual(result.numerical_dissipation_joule, scale)
            self.assertLessEqual(abs(result.energy_balance_residual_joule
                + result.numerical_dissipation_joule), scale)
            network.commit_preview(result)
            self.assertEqual(network.volts, result.observation.end_volts)
            with self.assertRaises(ValueError):
                network.commit_preview(result)
            results.append(result)
        for attribute in ("end_volts", "mean_volts"):
            np.testing.assert_allclose(getattr(results[0].observation, attribute),
                getattr(results[1].observation, attribute), rtol=1e-7, atol=2e-10)
        np.testing.assert_allclose([getattr(results[0], name) for name in WORK_FIELDS],
            [getattr(results[1], name) for name in WORK_FIELDS], rtol=1e-7, atol=2e-11)

    def test_unsupported_groups_fall_back_and_failed_attempt_does_not_commit(self):
        reference, arguments, *_ = generic_fixture()
        arguments["controls"] = replace(arguments["controls"], integration_method="radau_iia_2")
        expected = reference.advance(**arguments)
        network, fallback_arguments, *_ = generic_fixture()
        fallback_arguments["controls"] = replace(arguments["controls"], radau_preconditioner="contact_block")
        actual = network.advance(**fallback_arguments)
        self.assertEqual(actual.radau_preconditioner, "stage_jacobi")
        np.testing.assert_array_equal(actual.observation.end_volts, expected.observation.end_volts)
        np.testing.assert_array_equal(actual.observation.mean_volts, expected.observation.mean_volts)
        np.testing.assert_array_equal([getattr(actual, name) for name in WORK_FIELDS],
            [getattr(expected, name) for name in WORK_FIELDS])
        for changes, message in ((dict(max_workspace_bytes=1), "workspace"),
                                 (dict(max_operator_products=1), "operator-product")):
            network, arguments, _, _ = contact_problem("contact_block")
            arguments["controls"] = replace(arguments["controls"], **changes)
            before = network.volts
            with self.subTest(changes=changes), self.assertRaisesRegex(SolverConvergenceError, message):
                network.advance(**arguments)
            self.assertEqual(network.volts, before)


if __name__ == "__main__":
    unittest.main()
