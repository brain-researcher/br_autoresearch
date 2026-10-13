"""Full-array external-port identities; fixture constants are not device estimates."""

import math
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from daq_mux_hold import (DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend,
                          PassiveMuxHoldGroup)
from test_cpe_array import advance, make_array


REFERENCE = "explicit_fixture_reference"


def group(*, source_indices=(0,), initial_hold_volts=0.3, leak=0.05):
    return PassiveMuxHoldGroup(
        adc_id="fixture_adc", source_indices=source_indices,
        hold_capacitance_farad=0.7, initial_hold_volts=initial_hold_volts,
        track_conductance_siemens=(0.8,) if source_indices else (),
        off_conductance_siemens=(0.03,) if source_indices else (),
        hold_leak_conductance_siemens=leak,
        external_inputs=(ExternalVoltageInput("G1", 0.6, 0.02),))


def frontend(specification):
    return PassiveDAQFrontend((specification,), "additional_fixed_shunt_excludes_frontend")


def drive(selection, voltage, reference=REFERENCE):
    return {"fixture_adc": DAQGroupDrive(selection, (("G1", voltage),), reference)}


def reduced_interval(start, selection, voltage, duration):
    """Independent symmetry reduction: 64 identical pads, source 0, ADC hold.

    The production computation still advances all 4161 states. Dense moment
    integration also computes resistor loss independently of reported work.
    """
    capacitance = np.array([64 * 3, 1, 0.7])
    neural_g = 0.8 if selection == 0 else 0.03
    external_g = 0.6 if selection == "G1" else 0.02
    conductance = np.diag([0, 0.03, 0.05 + external_g])
    loss = np.diag([0.0, 0.03, 0.05, 0.0])
    for left, right, value in ((0, 1, 64 * 0.2), (1, 2, neural_g)):
        edge = np.zeros(3)
        edge[left], edge[right] = 1, -1
        conductance += value * np.outer(edge, edge)
        loss[:3, :3] += value * np.outer(edge, edge)
    external_drop = np.array([0, 0, 1, -voltage])
    loss += external_g * np.outer(external_drop, external_drop)
    generator = np.zeros((4, 4))
    generator[:3, :3] = -conductance / capacitance[:, None]
    generator[2, 3] = external_g * voltage / capacitance[2]
    augmented = np.zeros((7, 7))
    augmented[:4, :4] = generator
    augmented[4:, :3] = np.eye(3)
    initial = np.r_[start, 1]
    result = expm(duration * augmented) @ np.r_[initial, np.zeros(3)]
    moment = np.zeros((17, 17))
    moment[:16, :16] = np.kron(generator, np.eye(4)) + np.kron(np.eye(4), generator)
    moment[16, :16] = loss.ravel()
    dissipated = (expm(duration * moment) @ np.r_[np.kron(initial, initial), 0])[16]
    return result[:3], result[4:] / duration, dissipated, external_g


class CPEExternalInputTests(unittest.TestCase):
    def test_shared_adc_switching_matches_full_array_reduction_and_energy_balance(self):
        initial_sources = np.zeros(64)
        initial_sources[0] = 1
        array = make_array(
            daq_frontend=frontend(group()), initial_source_volts=initial_sources,
            tft_on_conductance_siemens=np.full((64, 64), 0.2),
            tft_off_conductance_siemens=np.full((64, 64), 0.2),
            source_load_conductance_siemens=np.full(64, 0.03))
        self.assertEqual((array.resources.base_states, array.resources.total_states,
                          array.resources.daq_hold_states, array.resources.daq_switch_branches),
                         (4160, 4161, 1, 2))
        expected = np.array([0, 1, 0.3])
        for selection, voltage, row in (("G1", 0.9, 0), (0, -0.2, 31), (None, 0.4, 63)):
            duration = 0.125
            end, mean, dissipated, external_g = reduced_interval(expected, selection, voltage, duration)
            result = advance(array, gate_row=row, duration_seconds=duration,
                             field_volts=np.zeros((64, 64)),
                             daq_selected_sources=drive(selection, voltage))
            kernel = result.kernel_interval.observation
            indices = [0, 4096, 4160]
            np.testing.assert_allclose(np.asarray(kernel.end_volts)[indices], end,
                                       rtol=2e-9, atol=2e-12)
            np.testing.assert_allclose(np.asarray(kernel.mean_volts)[indices], mean,
                                       rtol=2e-9, atol=2e-12)
            self.assertAlmostEqual(result.daq_hold_observations[0].mean_volts, mean[-1], places=10)
            work = duration * external_g * voltage * (voltage - mean[-1])
            self.assertAlmostEqual(result.daq_external_source_interval_work_joule, work, places=11)
            energy_change = result.physical_energy_end_joule - result.physical_energy_after_boundary_joule
            self.assertAlmostEqual(result.daq_external_source_interval_work_joule - energy_change,
                                   dissipated, places=10)
            self.assertEqual(result.field_source_interval_work_joule, 0)
            self.assertEqual(result.field_source_boundary_work_joule, 0)
            self.assertEqual(result.supplied_charge_boundary_work_joule, 0)
            self.assertEqual(result.physical_energy_after_boundary_joule,
                             result.physical_energy_before_boundary_joule)
            expected = end

    def test_external_only_hold_has_signed_work_off_loading_and_no_voltage_jump_charge(self):
        array = make_array(daq_frontend=frontend(group(source_indices=(), initial_hold_volts=2, leak=0)))
        expected_start = 2.0
        for selection, voltage, conductance in (("G1", 1.0, 0.6), (None, -0.25, 0.02)):
            duration, capacitance = 0.25, 0.7
            decay = conductance / capacitance
            difference = expected_start - voltage
            expected_end = voltage + difference * math.exp(-decay * duration)
            expected_mean = voltage + difference * -math.expm1(-decay * duration) / (decay * duration)
            loss = conductance * difference**2 * -math.expm1(-2 * decay * duration) / (2 * decay)
            result = advance(array, field_volts=np.zeros((64, 64)), duration_seconds=duration,
                             daq_selected_sources=drive(selection, voltage))
            observation = result.kernel_interval.observation
            self.assertEqual(observation.start_volts[-1], result.kernel_interval.boundary.transition.before_volts[-1])
            self.assertAlmostEqual(observation.start_volts[-1], expected_start, places=12)
            self.assertAlmostEqual(result.daq_hold_observations[0].end_volts, expected_end, places=12)
            self.assertAlmostEqual(result.daq_hold_observations[0].mean_volts, expected_mean, places=12)
            np.testing.assert_array_equal(np.asarray(observation.end_volts)[:4160], np.zeros(4160))
            work = duration * conductance * voltage * (voltage - expected_mean)
            self.assertAlmostEqual(result.daq_external_source_interval_work_joule, work, places=12)
            self.assertAlmostEqual(work - 0.5 * capacitance * (expected_end**2 - expected_start**2), loss, places=12)
            if selection == "G1":
                self.assertLess(result.daq_external_source_interval_work_joule, 0)
            self.assertEqual(result.supplied_charge_boundary_work_joule, 0)
            self.assertEqual(result.physical_energy_before_boundary_joule,
                             result.physical_energy_after_boundary_joule)
            expected_start = expected_end

    def test_reference_and_missing_off_voltage_refuse_before_preview_legacy_work_is_zero(self):
        array = make_array(daq_frontend=frontend(group()))
        before = array.volts, array.previous_field_volts
        bad = (drive("G1", 0.9, "different_reference"),
               {"fixture_adc": DAQGroupDrive(None, (), REFERENCE)})
        with patch.object(array._network, "advance", wraps=array._network.advance) as preview:
            for selection in bad:
                with self.subTest(selection=selection), self.assertRaises(ValueError):
                    advance(array, daq_selected_sources=selection)
                self.assertEqual((array.volts, array.previous_field_volts), before)
            preview.assert_not_called()
        plain = make_array()
        result = advance(plain)
        self.assertEqual(result.daq_external_source_interval_work_joule, 0)
        self.assertEqual(result.daq_hold_observations, ())
        self.assertEqual(plain.resources.total_states, 4160)


if __name__ == "__main__":
    unittest.main()
