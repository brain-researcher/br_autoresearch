"""Focused sparse/dense and full-lattice identities with non-device fixtures."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from coupled_rc import PassiveRCNetwork
from sparse_array_rc import (NeurocamPassiveArray, SolverControls, SolverConvergenceError,
                             SparsePassiveRCNetwork, passive_matrix)


def controls(**changes):
    values = dict(relative_tolerance=1e-10, voltage_atol=1e-13, charge_atol_coulomb=1e-22,
                  max_refinements=2, max_operator_products=20000)
    values.update(changes)
    return SolverControls(**values)


class SparseArrayRCTests(unittest.TestCase):
    def test_zero_reference_capacitors_require_grounded_positive_capacitor_components(self):
        network = SparsePassiveRCNetwork(state_ids=("grounded", "internal", "separate"),
                                         initial_volts=[1, 2, 3],
                                         capacitance_to_reference_farad=[2, 0, 4],
                                         mutual_capacitance_edges=[(0, 1, 0.5)])
        self.assertEqual(network.volts, (1, 2, 3))
        transition = network.switch_capacitance(capacitance_to_reference_farad=[4, 0, 8],
                                                 mutual_capacitance_edges=[(0, 1, 1.0)],
                                                 injected_charge_coulomb=[0, 0, 0], controls=controls())
        np.testing.assert_allclose(transition.transition.after_volts, [0.5, 1, 1.5], atol=1e-14)
        before = network.volts
        for shunts, edges in (([0, 0, 4], [(0, 1, 0.5)]),
                              ([2, 0, 4], [(0, 1, 0.0)]), ([2, 0, 4], [])):
            with self.subTest(shunts=shunts, edges=edges), self.assertRaises(ValueError):
                network.switch_capacitance(capacitance_to_reference_farad=shunts,
                                             mutual_capacitance_edges=edges,
                                             injected_charge_coulomb=[0, 0, 0], controls=controls())
            self.assertEqual(network.volts, before)

    def network(self):
        return SparsePassiveRCNetwork(state_ids=("a", "b", "source"), initial_volts=[0.3, -0.2, 0.1],
                                      capacitance_to_reference_farad=[2.0, 1.0, 3.0],
                                      mutual_capacitance_edges=[(0, 1, 0.2), (1, 2, 0.4)])

    def advance(self, network, **changes):
        arguments = dict(duration_seconds=0.8, conductance_to_reference_siemens=[0.2, 0.0, 0.3],
                         resistor_edges=[(0, 1, 0.7), (1, 2, 0.5)], drive_current_amp=[0.4, -0.1, 0.0],
                         observation_matrix=csr_matrix([[1.0, 0.0, -0.5], [0.0, 0.0, 1.0]]),
                         feedthrough_volts=[0.2, -0.1], injected_charge_coulomb=[0.0, 0.0, 0.0],
                         controls=controls())
        arguments.update(changes)
        return network.advance(**arguments)

    def test_sparse_coupled_response_matches_validated_dense_kernel(self):
        capacitance = passive_matrix(shunts=[2, 1, 3], edges=[(0, 1, 0.2), (1, 2, 0.4)],
                                     positive_shunts=True).toarray()
        conductance = passive_matrix(shunts=[0.2, 0, 0.3], edges=[(0, 1, 0.7), (1, 2, 0.5)],
                                     positive_shunts=False).toarray()
        dense = PassiveRCNetwork(state_ids=("a", "b", "source"), initial_volts=[0.3, -0.2, 0.1],
                                 capacitance_farad=capacitance)
        expected = dense.advance(duration_seconds=0.8, conductance_siemens=conductance,
                                 input_conductance_siemens=[[0.4], [-0.1], [0.0]], input_volts=[1.0],
                                 observation_matrix=[[1.0, 0.0, -0.5], [0.0, 0.0, 1.0]],
                                 feedthrough_matrix=[[0.2], [-0.1]])
        actual = self.advance(self.network())
        for name in ("end_volts", "mean_volts", "output_mean_volts", "output_end_volts",
                     "end_charge_coulomb", "end_energy_joule", "input_work_joule"):
            np.testing.assert_allclose(getattr(actual.observation, name), getattr(expected, name),
                                       rtol=1e-11, atol=1e-13, err_msg=name)
        self.assertLessEqual(actual.maximum_scaled_partition_error, 1.0)
        self.assertGreater(actual.operator_products, 0)
        self.assertEqual(actual.subintervals, 2)

    def test_variable_storage_conserves_every_node_charge(self):
        network = self.network()
        old_c = passive_matrix(shunts=[2, 1, 3], edges=[(0, 1, 0.2), (1, 2, 0.4)], positive_shunts=True)
        charge = old_c @ np.array(network.volts)
        result = network.switch_capacitance(capacitance_to_reference_farad=[4, 2, 6],
                                            mutual_capacitance_edges=[(0, 1, 0.4), (1, 2, 0.8)],
                                            injected_charge_coulomb=[0, 0, 0], controls=controls())
        np.testing.assert_allclose(result.transition.after_volts, [0.15, -0.1, 0.05], atol=1e-14)
        np.testing.assert_allclose(result.transition.after_charge_coulomb, charge, atol=1e-14)
        self.assertAlmostEqual(result.transition.after_energy_joule, result.transition.before_energy_joule / 2)
        self.assertLessEqual(result.maximum_scaled_charge_residual, 1.0)
        injected = np.array([0.1, -0.2, 0.3])
        driven = self.advance(network, injected_charge_coulomb=injected)
        np.testing.assert_allclose(driven.boundary.transition.after_charge_coulomb, charge + injected, atol=1e-14)

    def test_work_limit_and_invalid_branches_cannot_partially_apply_switch(self):
        network = self.network()
        original = network.volts
        with self.assertRaises(SolverConvergenceError):
            self.advance(network, injected_charge_coulomb=[0.1, 0.2, 0.3],
                         controls=controls(max_operator_products=1))
        self.assertEqual(network.volts, original)
        for changes in ({"duration_seconds": np.complex128(0.8 + 1j)},
                        {"resistor_edges": [(0, 1, -1e-16)]},
                        {"resistor_edges": [(0, 3, 1.0)]},
                        {"conductance_to_reference_siemens": [-1e-16, 0, 0]},
                        {"observation_matrix": csr_matrix([[1j, 0, 0]])}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.advance(network, **changes)
            self.assertEqual(network.volts, original)
        with self.assertRaises(ValueError):
            network.switch_capacitance(capacitance_to_reference_farad=[1, 0, 1],
                                        mutual_capacitance_edges=[], injected_charge_coulomb=[0, 0, 0],
                                        controls=controls())
        self.assertEqual(network.volts, original)
        for name in ("relative_tolerance", "voltage_atol", "charge_atol_coulomb"):
            with self.subTest(tolerance=name), self.assertRaises(ValueError):
                controls(**{name: np.complex128(1e-10 + 1j)})

    def test_zero_conductance_means_and_work_are_analytic(self):
        network = SparsePassiveRCNetwork(state_ids=("x", "y"), initial_volts=[1, 2],
                                         capacitance_to_reference_farad=[2, 4], mutual_capacitance_edges=[])
        result = network.advance(duration_seconds=0.6, conductance_to_reference_siemens=[0, 0],
                                 resistor_edges=[], drive_current_amp=[1, -2], observation_matrix=csr_matrix(np.eye(2)),
                                 feedthrough_volts=[0, 0], injected_charge_coulomb=[0, 0], controls=controls())
        np.testing.assert_allclose(result.observation.end_volts, [1.3, 1.7], rtol=1e-12)
        np.testing.assert_allclose(result.observation.mean_volts, [1.15, 1.85], rtol=1e-12)
        self.assertAlmostEqual(result.observation.input_work_joule,
                               result.observation.end_energy_joule - result.observation.start_energy_joule)

    def test_full_4096_pad_array_retains_off_gate_and_unread_states(self):
        # Arithmetic fixtures, deliberately not NeuroCam parameter estimates.
        pad_initial = np.zeros((64, 64))
        pad_initial[0] = 1e-3
        horizontal_edges = [(64 * row + col, 64 * row + col + 1, 1e-10)
                            for row in range(64) for col in range(63)]
        array = NeurocamPassiveArray(
            initial_pad_volts=pad_initial, initial_source_volts=np.zeros(64),
            pad_capacitance_farad=np.full((64, 64), 1e-9), source_capacitance_farad=np.full(64, 1e-8),
            mutual_capacitance_edges=horizontal_edges,
            tft_on_conductance_siemens=np.full((64, 64), 2e-6),
            tft_off_conductance_siemens=np.full((64, 64), 1e-7),
            pad_drive_conductance_siemens=np.zeros((64, 64)), source_load_conductance_siemens=np.zeros(64),
            other_resistor_edges=[],
        )
        self.assertEqual(len(array.network.state_ids), 4160)
        self.assertEqual(array.network.state_ids[4095], "pad_63_63")
        self.assertLess(array.network.capacitance_nnz, 20000)
        first = array.advance_row(gate_row=0, duration_seconds=62.5e-6, pad_drive_volts=np.zeros((64, 64)),
                                  observed_sources=tuple(range(64)), injected_charge_coulomb=np.zeros(4160),
                                  controls=controls())
        charge_start = sum(first.observation.start_charge_coulomb)
        np.testing.assert_allclose(sum(first.observation.end_charge_coulomb), charge_start, rtol=1e-10)
        self.assertLess(first.observation.end_energy_joule, first.observation.start_energy_joule)
        self.assertGreater(first.observation.end_volts[4095], 0.0)  # Off-gate pad retained.
        self.assertEqual(len(first.observation.output_mean_volts), 64)
        self.assertLessEqual(first.maximum_scaled_partition_error, 1.0)
        stored = np.array(array.network.volts)
        switch = array.network.switch_capacitance(
            capacitance_to_reference_farad=np.concatenate((np.full(4096, 2e-9), np.full(64, 2e-8))),
            mutual_capacitance_edges=[(i, j, 2 * value) for i, j, value in horizontal_edges],
            injected_charge_coulomb=np.zeros(4160), controls=controls())
        np.testing.assert_allclose(switch.transition.after_volts, stored / 2, rtol=1e-10, atol=1e-15)
        injected = np.zeros(4160)
        injected[-1] = 1e-12
        second = array.advance_row(gate_row=63, duration_seconds=62.5e-6, pad_drive_volts=np.zeros((64, 64)),
                                   observed_sources=(7, 19), injected_charge_coulomb=injected, controls=controls())
        self.assertEqual(len(second.observation.end_volts), 4160)
        self.assertEqual(len(second.observation.output_mean_volts), 2)
        np.testing.assert_allclose(sum(second.observation.start_charge_coulomb), charge_start + 1e-12, rtol=1e-10)
        np.testing.assert_allclose(sum(second.observation.end_charge_coulomb), charge_start + 1e-12, rtol=1e-10)
        self.assertGreater(second.observation.end_volts[4095], 0.0)
        self.assertLessEqual(second.maximum_scaled_partition_error, 1.0)
        print("Full_lattice_states=4160", f"capacitance_nnz={array.network.capacitance_nnz}",
              f"operator_products={first.operator_products + second.operator_products}",
              f"maximum_scaled_partition_error={max(first.maximum_scaled_partition_error, second.maximum_scaled_partition_error):.6g}")


if __name__ == "__main__":
    unittest.main()
