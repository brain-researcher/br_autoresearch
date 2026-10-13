"""Joint passive-array/DAQ identities; constants are not device estimates."""

from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm

from acquisition_readout import AcquisitionReadout, Observation, QuantizerTransfer, RawConversion, ReturnedSample
from cpe_array import AssemblyLimits
from daq_mux_hold import PassiveDAQFrontend, PassiveMuxHoldGroup
from test_cpe_array import advance, make_array


def group(**changes):
    values = dict(adc_id="fixture_adc_a", source_indices=(0, 1), hold_capacitance_farad=0.7,
                  initial_hold_volts=0.3, track_conductance_siemens=(0.8, 0.6),
                  off_conductance_siemens=(0.03, 0.02), hold_leak_conductance_siemens=0.05)
    values.update(changes)
    return PassiveMuxHoldGroup(**values)


def frontend(*groups):
    return PassiveDAQFrontend(groups, "additional_fixed_shunt_excludes_frontend")


def reduced_interval(start, selected, duration):
    """Independent five-node reduction for two uniform source columns + hold.

    Each column's 64 pad voltages are identical in this fixture, so summing
    their actual pad/contact C and TFT G yields one aggregate pad node. This
    is a symmetry reduction of the test circuit, never the production array.
    """
    capacitance = np.array([64 * 3, 1, 64 * 3, 1, 0.7])
    conductance = np.diag([0, 0.03, 0, 0.03, 0.05])
    paths = ((0, 1, 64 * 0.2), (2, 3, 64 * 0.2),
             (1, 4, 0.8 if selected == 0 else 0.03),
             (3, 4, 0.6 if selected == 1 else 0.02))
    for a, b, value in paths:
        conductance[a, a] += value
        conductance[b, b] += value
        conductance[a, b] -= value
        conductance[b, a] -= value
    augmented = np.zeros((10, 10))
    augmented[:5, :5] = -conductance / capacitance[:, None]
    augmented[5:, :5] = np.eye(5)
    result = expm(duration * augmented) @ np.r_[start, np.zeros(5)]
    return result[:5], result[5:] / duration


class DAQMuxHoldTests(unittest.TestCase):
    def test_joint_mux_history_matches_dense_reduction_and_whole_aperture(self):
        initial_sources = np.zeros(64)
        initial_sources[:2] = [1, -0.25]
        # Equal on/off TFT fixture values allow the independent column reduction;
        # DAQ switches still change independently of the declared row changes.
        array = make_array(daq_frontend=frontend(group()), initial_source_volts=initial_sources,
                           tft_on_conductance_siemens=np.full((64, 64), 0.2),
                           tft_off_conductance_siemens=np.full((64, 64), 0.2),
                           source_load_conductance_siemens=np.full(64, 0.03))
        self.assertEqual((array.resources.total_states, array.resources.daq_hold_states,
                          array.resources.daq_switch_branches), (4161, 1, 2))
        transfer = QuantizerTransfer(-2, 2, (0,), (-1, 1))
        readout = AcquisitionReadout(
            conversions=[RawConversion("raw", "fixture_adc_a", "explicit_hold_input", "signal",
                                       0, 0.5, 0.5, 0.25, transfer, True)],
            returned_samples=[ReturnedSample("word", ("raw",), (1,), 1, 0.25, 0.5, None)],
            observations=[Observation("pixel", "fixture_site", "pixel", ("word",), (1,), 1,
                                      0.25, 0.75, 8, None)],
            max_conversions=1, max_returned_samples=1, max_observations=1, max_interval_updates=4)
        expected = np.array([0, 1, 0, -0.25, 0.3])
        expected_integral = 0.0
        start = 0.0
        selected_states = [0, 4096, 1, 4097, 4160]
        # Mux-only switch, then row-only switch, then hold-only transition:
        # there is no assumed lock between the two clocks.
        for selection, duration, row in ((0, 0.125, 0), (1, 0.125, 0),
                                         (1, 0.125, 31), (None, 0.125, 31)):
            expected_end, expected_mean = reduced_interval(expected, selection, duration)
            result = advance(array, gate_row=row, duration_seconds=duration, field_volts=np.zeros((64, 64)),
                             observed_sources=(0, 1), daq_selected_sources={"fixture_adc_a": selection})
            np.testing.assert_allclose(np.array(result.kernel_interval.observation.end_volts)[selected_states],
                                       expected_end, rtol=2e-9, atol=2e-12)
            np.testing.assert_allclose(np.array(result.kernel_interval.observation.mean_volts)[selected_states],
                                       expected_mean, rtol=2e-9, atol=2e-12)
            held, = result.daq_hold_observations
            self.assertEqual((held.adc_id, held.state_id), ("fixture_adc_a", "daq_hold_fixture_adc_a"))
            self.assertAlmostEqual(held.mean_volts, expected_mean[-1], places=10)
            self.assertEqual(len(result.kernel_interval.observation.output_mean_volts), 2)
            # Explicit caller binding only; the physical ADC ID is not inferred
            # to be an observation's site or a readout input identifier.
            readout.add_interval(start=start, end=start + duration,
                                 mean_volts_by_input={"explicit_hold_input": held.mean_volts})
            expected_integral += duration * expected_mean[-1]
            expected = expected_end
            start += duration
        self.assertEqual(readout.observations_available_by(0.5), ())
        report = readout.finalize(finished_at=0.75, acquisition_start=0, acquisition_end=0.5)
        self.assertAlmostEqual(report["raw_values"][0].aperture_mean_volts, expected_integral / 0.5, places=10)
        self.assertEqual(report["actual_internal_conversions"], 1)

    def test_all_off_hold_charge_storage_and_bidirectional_source_loading(self):
        specification = frontend(group(source_indices=(3,), track_conductance_siemens=(0.8,),
                                       off_conductance_siemens=(0,), hold_leak_conductance_siemens=0,
                                       initial_hold_volts=0.2))
        array = make_array(daq_frontend=specification)
        charge = np.zeros(len(array.state_ids))
        charge[-1] = 0.14
        held = advance(array, field_volts=np.zeros((64, 64)), duration_seconds=0.125,
                       injected_charge_coulomb=charge, daq_selected_sources={"fixture_adc_a": None})
        self.assertAlmostEqual(held.daq_hold_observations[0].mean_volts, 0.4, places=13)
        self.assertAlmostEqual(held.daq_hold_observations[0].end_volts, 0.4, places=13)
        self.assertAlmostEqual(held.physical_energy_before_boundary_joule, 0.5 * 0.7 * 0.2**2, places=13)
        self.assertAlmostEqual(held.physical_energy_after_boundary_joule, 0.5 * 0.7 * 0.4**2, places=13)
        self.assertAlmostEqual(held.supplied_charge_boundary_work_joule, 0.14 * (0.2 + 0.4) / 2, places=13)
        self.assertAlmostEqual(held.physical_energy_end_joule, held.physical_energy_after_boundary_joule, places=13)
        self.assertEqual(held.field_source_interval_work_joule, 0)
        tracked = advance(array, gate_row=1, observed_sources=(3,), duration_seconds=0.25,
                          field_volts=np.zeros((64, 64)), daq_selected_sources={"fixture_adc_a": 3})
        self.assertGreater(tracked.kernel_interval.observation.end_volts[4099], 0)
        self.assertLess(tracked.daq_hold_observations[0].end_volts, 0.4)
        self.assertLess(tracked.physical_energy_end_joule, tracked.physical_energy_after_boundary_joule)
        self.assertAlmostEqual(tracked.physical_energy_before_boundary_joule, held.physical_energy_end_joule, places=13)
        retained = advance(array, gate_row=63, field_volts=np.zeros((64, 64)), duration_seconds=0.125,
                           daq_selected_sources={"fixture_adc_a": None})
        self.assertAlmostEqual(retained.daq_hold_observations[0].mean_volts,
                               tracked.daq_hold_observations[0].end_volts, places=13)

    def test_mapping_load_ownership_limits_and_invalid_selection_refuse_before_commit(self):
        specification = frontend(group())
        with self.assertRaises(ValueError):
            PassiveDAQFrontend((group(),), "load_already_in_source_shunt")
        with self.assertRaises(ValueError):
            frontend(group(), group())
        for changes in ({"hold_capacitance_farad": 0}, {"off_conductance_siemens": (-0.1, 0)},
                        {"track_conductance_siemens": (0, 1)}, {"initial_hold_volts": float("nan")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                group(**changes)
        with self.assertRaises(ValueError):
            make_array(daq_frontend=frontend(replace(group(), source_indices=(0, 64))))
        with self.assertRaises(ValueError):
            make_array(daq_frontend=specification, limits=AssemblyLimits(4160, 100000, 100000, 64 * 1024**2))
        array = make_array(daq_frontend=specification)
        saved = array.volts, array.previous_field_volts
        for selection in (None, {}, {"fixture_adc_a": 2}, {"fixture_adc_a": True},
                          {"fixture_adc_a": 0, "unknown": None}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                advance(array, daq_selected_sources=selection)
            self.assertEqual((array.volts, array.previous_field_volts), saved)
        plain = make_array(daq_frontend=None)
        self.assertEqual((plain.resources.total_states, plain.resources.daq_hold_states,
                          plain.resources.daq_switch_branches), (4160, 0, 0))
        result = advance(plain, field_volts=np.zeros((64, 64)))
        self.assertEqual(result.daq_hold_observations, ())
        self.assertEqual(len(result.kernel_interval.observation.output_mean_volts), 1)
        with self.assertRaises(ValueError):
            advance(plain, daq_selected_sources={"fixture_adc_a": None})


if __name__ == "__main__":
    unittest.main()
