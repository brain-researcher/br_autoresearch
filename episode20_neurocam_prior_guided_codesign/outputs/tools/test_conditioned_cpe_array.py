"""Joint full-array conditioning seams; explicit fixture values, no device fit."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from analog_conditioning import AnalogConditioning, SourceConditioning
from array_acquisition import (AcquisitionLimits, ArrayAcquisition, ArrayInputBinding,
                               PhysicalInterval)
from conditioned_array_solver import ConditionedRCNetwork, ConditionedSolverControls
from cpe_array import AssemblyLimits
from daq_mux_hold import (DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend,
                          PassiveMuxHoldGroup)
from sparse_array_rc import SolverControls, SparsePassiveRCNetwork, SolverConvergenceError
from test_cpe_array import advance, make_array


REF = "explicit_fixture_reference"
ADC = "conditioned_fixture_0"
SELECTED = np.array([0, 4096, 4160, 4224, 4288])


def channel(**changes):
    values = dict(gain=1.2, offset_volts=0., drive_lower_volts=-10., drive_upper_volts=10.,
        input_capacitance_farad=.5, input_conductance_siemens=.05,
        buffer_capacitance_farad=.8, buffer_conductance_siemens=.9,
        output_conductance_siemens=.6, filter_capacitance_farad=.7,
        filter_leak_conductance_siemens=.04, initial_buffer_volts=0., initial_filter_volts=0.)
    values.update(changes)
    return SourceConditioning(**values)


def conditioning(first=None):
    return AnalogConditioning((channel() if first is None else first,) + (channel(),) * 63,
        "additional_fixed_shunt_excludes_conditioning",
        "additional_fixed_capacitance_excludes_conditioning", REF)


def solver_controls(**changes):
    # The independent 3nV mean assertions require tighter accuracy and enough
    # finite midpoint refinement work; production controls remain separate.
    values = dict(relative_tolerance=2e-7, voltage_atol=1e-10, charge_atol_coulomb=1e-11,
        max_refinements=11, max_operator_products=100000,
        linear_relative_tolerance=1e-10, linear_charge_atol_coulomb=1e-13,
        energy_atol_joule=1e-10, max_newton_iterations=12, max_linear_iterations=20000,
        gmres_restart=24, max_workspace_bytes=32 * 1024 * 1024)
    values.update(changes)
    return ConditionedSolverControls(**values)


def fixture(*, first=None, initial_source=1., external=False):
    # This explicit fixture partition has all 64 sources and 24 physical ADCs;
    # source 0 alone uses ADC 0 so its independent reduction is five states.
    membership = [(0,)] + [tuple(range(1 + 3*i, 4 + 3*i)) for i in range(17)]
    membership += [tuple(range(52 + 2*i, 54 + 2*i)) for i in range(6)]
    groups = tuple(PassiveMuxHoldGroup(f"conditioned_fixture_{index}", sources,
        .7, 0., (.8,) * len(sources), (.03,) * len(sources), .05,
        (ExternalVoltageInput("G1", .6, .02),) if index == 0 and external else ())
        for index, sources in enumerate(membership))
    initial = np.zeros(64)
    initial[0] = initial_source
    circuit = make_array(analog_conditioning=conditioning(first),
        daq_frontend=PassiveDAQFrontend(groups, "additional_fixed_shunt_excludes_frontend"),
        initial_source_volts=initial, source_load_conductance_siemens=np.full(64, .03),
        tft_on_conductance_siemens=np.full((64, 64), .2),
        tft_off_conductance_siemens=np.full((64, 64), .2))
    return circuit


def selections(selected=0, external=None):
    values = {f"conditioned_fixture_{i}": None for i in range(24)}
    values[ADC] = selected if external is None else DAQGroupDrive(selected, (("G1", external),), REF)
    return values


def reference(start, specification, duration, *, slope, offset, selected=0, external=None):
    """Independent dense symmetry reduction plus quadratic port-work integrals.

    Coordinates: all 64 equivalent pads on source 0, source, buffer, filter,
    same ADC hold. The other 63 columns remain at exact zero in these fixtures.
    slope/offset specify the known linear or constantly saturated drive regime.
    """
    cap = np.array([192., 1. + specification.input_capacitance_farad,
                    specification.buffer_capacitance_farad, specification.filter_capacitance_farad, .7])
    gb = specification.buffer_conductance_siemens
    neural = .8 if selected == 0 else .03
    external_g = 0. if external is None else (.6 if selected == "G1" else .02)
    conductance = np.diag([0., .03 + specification.input_conductance_siemens,
                           gb, specification.filter_leak_conductance_siemens, .05 + external_g])
    for a, b, value in ((0, 1, 12.8), (2, 3, specification.output_conductance_siemens), (3, 4, neural)):
        edge = np.zeros(5)
        edge[a], edge[b] = 1., -1.
        conductance += value * np.outer(edge, edge)
    generator = np.zeros((6, 6))
    generator[:5, :5] = -conductance / cap[:, None]
    generator[2, 1] += gb * slope / cap[2]
    generator[2, 5] = gb * offset / cap[2]
    generator[4, 5] = external_g * (0. if external is None else external) / cap[4]
    augmented = np.zeros((11, 11))
    augmented[:6, :6] = generator
    augmented[6:, :5] = np.eye(5)
    initial = np.r_[start, 1.]
    result = expm(duration * augmented) @ np.r_[initial, np.zeros(5)]
    u, b, h, ext = (np.zeros(6) for _ in range(4))
    u[1], u[5], b[2], h[4] = slope, offset, 1., 1.
    ext[5] = 0. if external is None else external
    port = gb * np.outer(u, u - b)
    external_port = external_g * np.outer(ext, ext - h)
    drive_heat = gb * np.outer(u - b, u - b)
    heat = np.zeros((6, 6))
    heat[:5, :5] = conductance
    heat += drive_heat - gb * np.outer(b, b)
    heat += external_g * (np.outer(ext - h, ext - h) - np.outer(h, h))
    moments = np.zeros((40, 40))
    moments[:36, :36] = np.kron(generator, np.eye(6)) + np.kron(np.eye(6), generator)
    moments[36:] = np.column_stack((np.array([port.ravel(), external_port.ravel(),
                                             drive_heat.ravel(), heat.ravel()]), np.zeros((4, 4))))
    work = (expm(duration * moments) @ np.r_[np.kron(initial, initial), np.zeros(4)])[36:]
    return result[:5], result[6:] / duration, work


class ConditionedCPEArrayTests(unittest.TestCase):
    def test_full_array_joint_gain_filter_loading_hold_means_and_port_energy(self):
        array = fixture()
        self.assertIsInstance(array._network, ConditionedRCNetwork)
        self.assertEqual((array.resources.base_states, array.resources.total_states,
            array.resources.conditioning_stages, array.resources.conditioning_states,
            array.resources.daq_hold_states, array.resources.daq_switch_branches),
            (4160, 4312, 64, 128, 24, 64))
        self.assertEqual(array.resources.conditioning_drive_numeric_bytes, 7 * 64 * 8)
        self.assertEqual(array.state_ids[4160], "conditioning_buffer_00")
        self.assertEqual(array.state_ids[4224], "conditioning_filter_00")
        self.assertEqual(array.state_ids[4288], "daq_hold_" + ADC)
        duration = .04
        expected, mean, work = reference(np.array([0., 1., 0., 0., 0.]), channel(), duration,
                                          slope=1.2, offset=0.)
        result = advance(array, field_volts=np.zeros((64, 64)), duration_seconds=duration,
            controls=solver_controls(), check=lambda: None, daq_selected_sources=selections())
        actual = result.kernel_interval.observation
        np.testing.assert_allclose(np.asarray(actual.end_volts)[SELECTED], expected, rtol=3e-6, atol=3e-9)
        np.testing.assert_allclose(np.asarray(actual.mean_volts)[SELECTED], mean, rtol=3e-6, atol=3e-9)
        self.assertAlmostEqual(result.daq_hold_observations[0].mean_volts, mean[-1], delta=3e-9)
        self.assertAlmostEqual(actual.output_mean_volts[0], mean[1], delta=3e-9)
        self.assertAlmostEqual(result.controlled_voltage_source_interval_work_joule, work[0], delta=3e-8)
        self.assertAlmostEqual(result.conditioning_drive_branch_dissipation_joule, work[2], delta=3e-8)
        energy = result.physical_energy_end_joule - result.physical_energy_after_boundary_joule
        self.assertAlmostEqual(result.controlled_voltage_source_interval_work_joule - energy, work[3], delta=4e-8)
        self.assertEqual(result.field_source_interval_work_joule, 0.)
        self.assertEqual(result.daq_external_source_interval_work_joule, 0.)
        np.testing.assert_array_equal(np.asarray(actual.end_volts)[4289:], np.zeros(23))

    def test_saturated_loaded_recovery_and_same_adc_aperture_keep_state_and_external_work(self):
        first = channel(gain=0., offset_volts=2., drive_lower_volts=-.3, drive_upper_volts=.3,
                        initial_buffer_volts=1., initial_filter_volts=.4)
        array = fixture(first=first, initial_source=0., external=True)
        expected = np.array([0., 0., 1., .4, 0.])
        first_end, first_mean, first_work = reference(expected, first, .03, slope=0., offset=.3,
                                                      selected="G1", external=.8)
        second_end, second_mean, second_work = reference(first_end, first, .04, slope=0., offset=.3,
                                                         selected=0, external=-.2)
        intervals = (PhysicalInterval(0., .03, 0, np.zeros((64, 64)), REF,
                        selections("G1", .8), np.zeros(len(array.state_ids))),
                     PhysicalInterval(.03, .07, 63, np.zeros((64, 64)), REF,
                        selections(0, -.2), np.zeros(len(array.state_ids))))
        transfer = QuantizerTransfer(-2., 2., (0.,), (-1., 1.))
        raw = RawConversion("r", ADC, "held", "signal", 0., .07, .07, 0., transfer, True)
        sample = ReturnedSample("s", ("r",), (1.,), 1., 0., .07, None)
        observation = Observation("o", "fixture_pixel", "pixel", ("s",), (1.,), 1., 0., .09, 8, 0)
        calls, physical = [], []
        original = array.advance_row
        def capture(**kwargs):
            value = original(**kwargs)
            physical.append(value)
            return value
        with patch.object(array, "advance_row", side_effect=capture):
            report = ArrayAcquisition(array=array, physical_intervals=intervals,
                input_bindings=(ArrayInputBinding("held", ADC, "hold", None, ADC),),
                conversions=(raw,), returned_samples=(sample,), observations=(observation,),
                acquisition_window=(0., .07), final_publication_horizon=.09,
                controls=solver_controls(), limits=AcquisitionLimits(2, 4, 1, 1, 1, 1024 * 1024),
                check=lambda: calls.append(None)).run()
        self.assertTrue(calls)
        self.assertEqual(len(physical), 2)
        self.assertEqual(physical[1].kernel_interval.boundary.transition.before_volts,
                         physical[0].kernel_interval.observation.end_volts)
        self.assertGreater(array.volts[4160], .3)  # Internal target rails do not clamp the loaded capacitor.
        np.testing.assert_allclose(np.asarray(array.volts)[SELECTED], second_end, rtol=3e-6, atol=3e-9)
        expected_mean = (.03 * first_mean[-1] + .04 * second_mean[-1]) / .07
        self.assertAlmostEqual(report["raw_values"][0].aperture_mean_volts, expected_mean, delta=3e-9)
        self.assertEqual((report["actual_internal_conversions"], report["hardware_returned_samples"],
                          report["retained_pixel_observations"]), (1, 1, 1))
        self.assertAlmostEqual(sum(v.daq_external_source_interval_work_joule for v in physical),
                               first_work[1] + second_work[1], delta=3e-8)
        self.assertAlmostEqual(sum(v.controlled_voltage_source_interval_work_joule for v in physical),
                               first_work[0] + second_work[0], delta=3e-8)
        self.assertLess(physical[0].controlled_voltage_source_interval_work_joule, 0.)

    def test_bad_controls_reference_budget_or_accounting_cannot_commit_and_passive_is_unchanged(self):
        with self.assertRaisesRegex(ValueError, "at most 8"):
            SolverControls(relative_tolerance=2e-7, voltage_atol=1e-10,
                charge_atol_coulomb=1e-11, max_refinements=9, max_operator_products=100000)
        with self.assertRaisesRegex(ValueError, "at most 12"):
            solver_controls(max_refinements=13)
        array = fixture()
        saved = array.volts, array.previous_field_volts
        arguments = dict(daq_selected_sources=selections(), controls=solver_controls(), check=lambda: None)
        with patch.object(array._network, "advance", wraps=array._network.advance) as preview:
            for changes in ({"voltage_reference_id": "other"}, {"check": None}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    advance(array, **{**arguments, **changes})
            preview.assert_not_called()
        with self.assertRaises(SolverConvergenceError):
            advance(array, **{**arguments, "controls": solver_controls(max_operator_products=1)})
        self.assertEqual((array.volts, array.previous_field_volts), saved)
        with patch.object(array, "_physical_energy", side_effect=ValueError("accounting failure")):
            with self.assertRaisesRegex(ValueError, "accounting failure"):
                advance(array, **arguments)
        self.assertEqual((array.volts, array.previous_field_volts), saved)
        with self.assertRaisesRegex(ValueError, "reference"):
            make_array(analog_conditioning=replace(conditioning(), voltage_reference_id="other"))
        with self.assertRaisesRegex(ValueError, "assembly limit"):
            make_array(analog_conditioning=conditioning(), limits=AssemblyLimits(4287, 100000, 100000, 2**26))
        plain, explicit_none = make_array(), make_array(analog_conditioning=None)
        self.assertIs(type(plain._network), SparsePassiveRCNetwork)
        self.assertEqual(advance(plain), advance(explicit_none))
        self.assertEqual((plain.resources.conditioning_stages, plain.resources.conditioning_states,
                          plain.resources.conditioning_drive_numeric_bytes), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
