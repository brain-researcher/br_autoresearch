"""Focused outer-solver checks for the separately selected Radau integration.

These checks require the optional integration patch. They do not select a
production method, read saved worlds or launch a recovery/parameter sweep.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from array_acquisition import ArrayAcquisition
from conditioned_array_solver import ConditionedRCNetwork
from development_simulation import _AccountingArray
from radau_polynomial_forcing import RadauPolynomialCircuitForcing
from sparse_array_rc import SolverConvergenceError
from streaming_array_acquisition import StreamingArrayAcquisition
import test_cpe_array
from test_conditioned_array_solver import controls
from test_conditioned_cpe_array import solver_controls
from test_conditioned_radau_step import fixture as step_fixture
from test_cpe_polynomial_noise import SELECTED, WORK_FIELDS, fixture, polynomial, reference, run
from test_polynomial_acquisition import (eager_arguments, fixture as acquisition_fixture,
                                         provider, stream_arguments)


RATE, AMPLITUDE, CAPACITANCE = 2 * math.pi * 1e8, 1e-7, 1e-12
DURATION = 800. / RATE


def stiff_problem(selected):
    existing = step_fixture(source_c=CAPACITANCE, source_g=RATE * CAPACITANCE,
        gain=0., initial=(AMPLITUDE, 0.), selected_controls=selected)
    network = ConditionedRCNetwork(state_ids=tuple(f"v:{i}" for i in range(128)),
        initial_volts=existing["old"], capacitance_to_reference_farad=existing["capacitance"].diagonal(),
        mutual_capacitance_edges=())
    arguments = dict(duration_seconds=DURATION,
        conductance_to_reference_siemens=existing["passive_diagonal"], resistor_edges=(),
        drive_current_amp=np.zeros(128), observation_matrix=csr_matrix(
            (np.ones(2), (np.arange(2), np.array([0, 64]))), shape=(2, 128)),
        feedthrough_volts=np.zeros(2), injected_charge_coulomb=np.zeros(128),
        controls=selected, controlled_drives=existing["controlled_drives"], check=lambda: None)
    return network, arguments


def selected_controls(**changes):
    values = dict(relative_tolerance=1e-5, voltage_atol=1e-8,
        charge_atol_coulomb=1e-20, linear_relative_tolerance=1e-9,
        linear_charge_atol_coulomb=1e-22, energy_atol_joule=1e-18)
    values.update(changes)
    return controls(**values)


class ConditionedRadauIntegrationTests(unittest.TestCase):
    def test_optional_radau_resolves_stiff_decay_without_changing_default_midpoint(self):
        selected = selected_controls()
        default, default_args = stiff_problem(selected)
        explicit, explicit_args = stiff_problem(replace(selected, integration_method="midpoint"))
        original = default.advance(**default_args)
        unchanged = explicit.advance(**explicit_args)
        self.assertEqual(original, unchanged)
        self.assertEqual(original.integration_method, "midpoint")
        self.assertEqual(original.numerical_dissipation_joule, 0.)
        exact_end = AMPLITUDE * math.exp(-800.)
        # This is the diagnosed two/four-part alias, not a new parameter sweep.
        self.assertGreater(abs(original.observation.end_volts[0] - exact_end),
                           9 * selected.voltage_atol)
        network, arguments = stiff_problem(replace(selected, integration_method="radau_iia_2"))
        before = network.volts
        result = network.advance(**arguments, commit=False)
        self.assertEqual(network.volts, before)
        self.assertEqual(result.integration_method, "radau_iia_2")
        np.testing.assert_allclose(result.observation.end_volts[:64], exact_end,
                                   rtol=0., atol=selected.voltage_atol)
        np.testing.assert_allclose(result.observation.mean_volts[:64],
            AMPLITUDE * (-math.expm1(-800.)) / 800., rtol=0., atol=1e-12)
        np.testing.assert_array_equal(result.observation.end_volts[64:], np.zeros(64))
        raw_balance = (result.observation.end_energy_joule
                       - result.observation.start_energy_joule
                       - result.observation.input_work_joule + result.passive_norton_dissipation_joule)
        self.assertAlmostEqual(result.energy_balance_residual_joule, raw_balance, delta=1e-35)
        self.assertGreater(result.numerical_dissipation_joule, 10 * result.passive_norton_dissipation_joule)
        self.assertAlmostEqual(raw_balance + result.numerical_dissipation_joule, 0., delta=1e-30)
        self.assertLess(abs(raw_balance), selected.energy_atol_joule)
        network.commit_preview(result)
        self.assertEqual(network.volts, result.observation.end_volts)
        with self.assertRaises(ValueError):
            network.commit_preview(result)

    def test_numerical_loss_cannot_rescue_failed_raw_energy_acceptance(self):
        selected = selected_controls(integration_method="radau_iia_2", max_refinements=1,
                                     energy_atol_joule=1e-26)
        # The scalar stability function and quadrature give an independent
        # case where one/two-part agreement passes, but raw energy does not.
        def response(z):
            denominator = 1 + 2 * z / 3 + z * z / 6
            first, last = (1 + z / 3) / denominator, (1 - z / 3) / denominator
            heat = z * (.75 * first * first + .25 * last * last)
            return last, heat
        coarse_end, coarse_heat = response(800.)
        half_end, half_heat = response(400.)
        fine_end, fine_heat = half_end**2, half_heat * (1 + half_end**2)
        energy_unit = 64 * CAPACITANCE * AMPLITUDE**2
        self.assertLess(AMPLITUDE * abs(coarse_end - fine_end), selected.voltage_atol)
        self.assertLess(AMPLITUDE * abs(coarse_end - fine_end) / 800., selected.voltage_atol)
        self.assertLess(energy_unit * abs(coarse_heat - fine_heat), selected.energy_atol_joule)
        raw_balance = energy_unit * (.5 * (fine_end**2 - 1) + fine_heat)
        energy_scale = selected.energy_atol_joule + selected.relative_tolerance * energy_unit / 2
        self.assertGreater(abs(raw_balance), 10 * energy_scale)
        network, arguments = stiff_problem(selected)
        before = network.volts
        with self.assertRaisesRegex(SolverConvergenceError, "partition agreement|energy"):
            network.advance(**arguments)
        self.assertEqual(network.volts, before)

    def test_full_cpe_cubic_endpoint_binding_and_original_work_match_existing_reference(self):
        array, layout = fixture()
        noise = polynomial(layout)
        selected = solver_controls(integration_method="radau_iia_2", relative_tolerance=2e-6,
            voltage_atol=1e-9, charge_atol_coulomb=1e-10, energy_atol_joule=1e-9,
            max_operator_products=1_000_000, max_linear_iterations=100000)
        before = array.volts, array.previous_field_volts
        with self.assertRaisesRegex(SolverConvergenceError, "workspace"):
            run(array, noise, controls=replace(selected, max_workspace_bytes=1))
        self.assertEqual((array.volts, array.previous_field_volts), before)
        endpoint_queries = []
        ordinary_left = RadauPolynomialCircuitForcing.evaluate_left
        def record_left(bound, time):
            endpoint_queries.append((time, bound._duration, bound._polynomial))
            return ordinary_left(bound, time)
        with patch.object(RadauPolynomialCircuitForcing, "evaluate_left", new=record_left):
            result = run(array, noise, controls=selected)
        expected, expected_mean, expected_work = reference()
        kernel = result.kernel_interval
        np.testing.assert_allclose(np.asarray(kernel.observation.end_volts)[SELECTED], expected,
                                   rtol=1e-5, atol=3e-8)
        np.testing.assert_allclose(np.asarray(kernel.observation.mean_volts)[SELECTED], expected_mean,
                                   rtol=1e-5, atol=3e-8)
        np.testing.assert_allclose([getattr(result, name) for name in WORK_FIELDS], expected_work,
                                   rtol=2e-5, atol=3e-9)
        self.assertEqual(kernel.integration_method, "radau_iia_2")
        self.assertTrue(any(time == duration for time, duration, _ in endpoint_queries))
        self.assertTrue(all(bound is noise for _, _, bound in endpoint_queries))
        self.assertEqual(array.resources.total_states, 4314)
        self.assertEqual(len(result.daq_hold_observations), 24)

    def test_eager_and_streaming_retain_separate_returned_numerical_dissipation(self):
        def accounted_array(**parameters):
            return _AccountingArray(total_operator_limit=16_000_000, checkpoint=lambda: None, **parameters)
        with patch.object(test_cpe_array, "NeurocamCPEArray", new=accounted_array):
            eager, eager_layout, eager_windows = acquisition_fixture()
        eager_args = eager_arguments(eager, eager_windows, provider(eager_layout))
        eager_args["controls"] = replace(eager_args["controls"], integration_method="radau_iia_2")
        eager_intervals = []
        original_eager = eager.advance_row
        def capture_eager(**kwargs):
            result = original_eager(**kwargs)
            eager_intervals.append(result)
            return result
        with patch.object(eager, "advance_row", new=capture_eager):
            ArrayAcquisition(**eager_args).run()
        expected_dissipation = math.fsum(
            interval.kernel_interval.numerical_dissipation_joule for interval in eager_intervals)
        self.assertGreater(expected_dissipation, 0.)
        self.assertTrue(all(interval.kernel_interval.integration_method == "radau_iia_2"
                            for interval in eager_intervals))

        streamed, stream_layout, stream_windows = acquisition_fixture()
        stream_args = stream_arguments(streamed, stream_windows, provider(stream_layout))
        stream_args["controls"] = replace(stream_args["controls"], integration_method="radau_iia_2")
        stream_intervals = []
        original_stream = streamed.advance_row
        def capture_stream(**kwargs):
            result = original_stream(**kwargs)
            stream_intervals.append(result)
            return result
        runner = StreamingArrayAcquisition(**stream_args)
        with patch.object(streamed, "advance_row", new=capture_stream):
            runner.advance_window()
            partial = runner.accounting_summary["numerical_integration"]
            self.assertEqual(partial["returned_electrical_intervals"], len(stream_intervals))
            self.assertAlmostEqual(partial["numerical_dissipation_joule"], math.fsum(
                interval.kernel_interval.numerical_dissipation_joule for interval in stream_intervals), delta=1e-18)
            self.assertTrue(partial["complete_for_returned_intervals"])
            stream_report = runner.run()
        for report in (eager.accounting(), stream_report, runner.accounting_summary):
            numerical = report["numerical_integration"]
            self.assertEqual(numerical["integration_method"], "radau_iia_2")
            self.assertEqual(numerical["returned_electrical_intervals"], len(eager_intervals))
            self.assertTrue(numerical["complete_for_returned_intervals"])
            self.assertAlmostEqual(numerical["numerical_dissipation_joule"], expected_dissipation, delta=1e-18)
        np.testing.assert_array_equal(streamed.volts, eager.volts)
        self.assertNotIn("numerical_dissipation_joule", stream_report["physical_work"])
        for name in WORK_FIELDS:
            self.assertAlmostEqual(stream_report["physical_work"][name],
                math.fsum(getattr(interval, name) for interval in eager_intervals), delta=1e-14)


if __name__ == "__main__":
    unittest.main()
