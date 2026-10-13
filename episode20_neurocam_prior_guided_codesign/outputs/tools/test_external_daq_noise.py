"""External-switch current noise at its physical voltage terminal, not ground.

The explicit fixture keeps 4096 pads, 64 neural sources and 24 ADC holds.
Only ADC0 is excited; its unselected neural path is open, allowing an exact
one-state RC reference without reducing the production array computation.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from daq_mux_hold import (DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend,
                          PassiveMuxHoldGroup, external_daq_branch_id)
from physical_noise_ports import (ExternalDAQCurrentPort, NoiseLimits, NodalCurrentPort,
                                   PhysicalNoiseLayout)
from test_cpe_array import advance, make_array


ADC, INPUT, REF = "external_noise_adc_0", "G1", "explicit_fixture_reference"
HOLD = "daq_hold_" + ADC
BRANCH = external_daq_branch_id(ADC, INPUT)
PORT = ExternalDAQCurrentPort("fixture_external_eta", BRANCH, ADC, INPUT, HOLD,
                              "conditional_external_switch_current")
LIMITS = NoiseLimits(100000, 8 * 1024**2)


def fixture(*, off=.02):
    membership = [(0,)] + [tuple(range(1 + 3*i, 4 + 3*i)) for i in range(17)]
    membership += [tuple(range(52 + 2*i, 54 + 2*i)) for i in range(6)]
    groups = tuple(PassiveMuxHoldGroup(f"external_noise_adc_{index}", sources,
        .7, .3 if index == 0 else 0., (.8,) * len(sources),
        (0.,) * len(sources), .05,
        (ExternalVoltageInput(INPUT, .6, off),) if index == 0 else ())
        for index, sources in enumerate(membership))
    return make_array(daq_frontend=PassiveDAQFrontend(groups, "additional_fixed_shunt_excludes_frontend"))


def bind(array, *, ports=(PORT,), nodal=(), limits=LIMITS):
    return array.make_noise_layout(nodal_current_ports=nodal,
        contact_noise_role="fixture_contact_noise_explicitly_zero",
        amplifier_input_noise_role=None, excluded_retained_contact_components=(),
        external_daq_current_ports=ports, limits=limits)


def sample(layout, amplitude):
    values = np.zeros(layout.numeric_value_count)
    values[layout.port_ids.index(PORT.port_id)] = amplitude
    return layout.sample_from_values(values)


def selection(selected=INPUT, voltage=.9, reference=REF):
    values = {f"external_noise_adc_{index}": None for index in range(24)}
    values[ADC] = DAQGroupDrive(selected, ((INPUT, voltage),), reference)
    return values


def exact_hold(initial, *, duration, conductance, source_voltage, current):
    capacitance, leak = .7, .05
    rate = (conductance + leak) / capacitance
    equilibrium = (conductance * source_voltage + current) / (conductance + leak)
    delta = initial - equilibrium
    integral_exp = -math.expm1(-rate * duration) / rate
    integral_exp2 = -math.expm1(-2 * rate * duration) / (2 * rate)
    end = equilibrium + delta * math.exp(-rate * duration)
    mean = equilibrium + delta * integral_exp / duration
    integral_square = (equilibrium**2 * duration + 2 * equilibrium * delta * integral_exp
                       + delta**2 * integral_exp2)
    heat = ((conductance + leak) * integral_square
            - 2 * conductance * source_voltage * mean * duration
            + conductance * source_voltage**2 * duration)
    return end, mean, heat


class ExternalDAQNoiseTests(unittest.TestCase):
    def test_selected_and_off_noise_match_analytic_hold_and_signed_physical_work(self):
        array = fixture()
        layout = bind(array)
        self.assertEqual((array.resources.base_states, array.resources.daq_hold_states,
                          array.resources.total_states), (4160, 24, 4184))
        self.assertEqual(array.resources.daq_switch_branches, 65)  # Existing64 neural +one external path.
        self.assertEqual(layout.port_ids[-1], PORT.port_id)
        self.assertEqual(layout.port_units[-1], "A")
        self.assertEqual(layout.external_daq_branches[0].hold_state_id, HOLD)
        self.assertEqual(layout.external_daq_branches[0].branch_id, BRANCH)
        initial = .3
        for selected, conductance, eta, row in ((INPUT, .6, .08, 0), (None, .02, -.05, 63)):
            duration, voltage = .15, .9
            expected_end, expected_mean, heat = exact_hold(initial, duration=duration,
                conductance=conductance, source_voltage=voltage, current=eta)
            result = advance(array, gate_row=row, field_volts=np.zeros((64, 64)),
                duration_seconds=duration, daq_selected_sources=selection(selected, voltage),
                noise_sample=sample(layout, eta))
            hold = result.daq_hold_observations[0]
            self.assertEqual(hold.adc_id, ADC)
            self.assertEqual(hold.state_id, HOLD)
            self.assertAlmostEqual(hold.end_volts, expected_end, places=11)
            self.assertAlmostEqual(hold.mean_volts, expected_mean, places=11)
            self.assertAlmostEqual(result.kernel_interval.observation.start_volts[4160], initial, places=12)
            np.testing.assert_array_equal(np.asarray(array.volts)[:4160], np.zeros(4160))
            np.testing.assert_array_equal(np.asarray(array.volts)[4161:], np.zeros(23))
            noise_work = duration * eta * (expected_mean - voltage)
            ideal_work = duration * voltage * (conductance * (voltage - expected_mean) + eta)
            self.assertAlmostEqual(result.noise_current_source_interval_work_joule, noise_work, places=12)
            self.assertAlmostEqual(result.daq_external_source_interval_work_joule, ideal_work, places=12)
            # The source-work correction and noise terminal work together
            # account for the nodal eta*v term. Neither port is grounded.
            passive_ideal_work = duration * conductance * voltage * (voltage - expected_mean)
            self.assertAlmostEqual(ideal_work - passive_ideal_work + noise_work,
                                   duration * eta * expected_mean, places=12)
            energy_change = result.physical_energy_end_joule - result.physical_energy_after_boundary_joule
            self.assertAlmostEqual(ideal_work + noise_work - energy_change, heat, places=11)
            self.assertEqual(math.copysign(1., result.noise_current_source_interval_work_joule),
                             -math.copysign(1., eta))
            self.assertEqual(result.noise_access_voltage_source_interval_work_joule, 0.)
            self.assertEqual(result.field_source_interval_work_joule, 0.)
            self.assertEqual(result.field_source_boundary_work_joule, 0.)
            self.assertEqual(result.supplied_charge_boundary_work_joule, 0.)
            self.assertEqual(result.physical_energy_before_boundary_joule,
                             result.physical_energy_after_boundary_joule)
            initial = expected_end

    def test_zero_off_path_and_reference_refuse_during_interval_preflight(self):
        array = fixture(off=0.)
        layout = bind(array)
        active = sample(layout, .1)
        saved = array.volts, array.previous_field_volts
        with patch.object(array._network, "advance", wraps=array._network.advance) as preview:
            for choices in (selection(None), selection(0), selection(reference="wrong_reference")):
                with self.subTest(choices=choices), self.assertRaises(ValueError):
                    array.validate_noise_interval(active, daq_selected_sources=choices)
                with self.assertRaises(ValueError):
                    advance(array, daq_selected_sources=choices, noise_sample=active)
                self.assertEqual((array.volts, array.previous_field_volts), saved)
            preview.assert_not_called()
        # Zero amplitudes are explicit for zero conductance. No unseen current
        # is added simply because the port remains in the immutable spectrum.
        zero = sample(layout, 0.)
        array.validate_noise_interval(zero, daq_selected_sources=selection(None))
        array.validate_noise_interval(active, daq_selected_sources=selection())
        context = array.daq_frontend.external_branch_drives(selected_sources=selection(None),
            first_hold_state=4160, voltage_reference_id=REF)
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].conductance_siemens, 0.)
        with self.assertRaises(ValueError):
            layout.validate_external_branch_sample(zero, external_branch_drives=(replace(
                context[0], voltage_reference_id="wrong_reference"),))
        result = advance(array, field_volts=np.zeros((64, 64)), duration_seconds=.1,
            daq_selected_sources=selection(None), noise_sample=zero)
        self.assertAlmostEqual(result.daq_hold_observations[0].end_volts,
                               .3 * math.exp(-.05 * .1 / .7), places=12)
        self.assertEqual(result.daq_external_source_interval_work_joule, 0.)
        self.assertEqual(result.noise_current_source_interval_work_joule, 0.)

    def test_exact_inventory_ownership_limits_atomic_accounting_and_legacy_none(self):
        array = fixture()
        original_resources = array.resources
        with self.assertRaisesRegex(ValueError, "canonical"):
            replace(PORT, branch_id="owner_alias")
        for ports, nodal in (((PORT, replace(PORT, port_id="duplicate")), ()),
                             ((replace(PORT, hold_state_id="source_00"),), ()),
                             ((PORT,), (NodalCurrentPort("ground_alias", HOLD, None,
                                "conditional_current", BRANCH),))):
            with self.subTest(ports=ports), self.assertRaises(ValueError):
                bind(array, ports=ports, nodal=nodal)
            self.assertIsNone(array.physical_noise_layout)
            self.assertEqual(array.resources, original_resources)
        with self.assertRaisesRegex(ValueError, "inventory"):
            PhysicalNoiseLayout(state_ids=array.state_ids, voltage_reference_id=REF,
                nodal_current_ports=(), contact_stars=array._contact_noise_stars,
                contact_ownership="all_eliminated_star_branch_and_access_noise_owned_here",
                contact_noise_role="conditional_fixture", amplifier_input_noise_role=None,
                excluded_retained_contact_components=(), limits=LIMITS,
                external_daq_current_ports=(PORT,))
        # Same inventory, one additional original amplitude and one cached
        # destination index: exactly16 retained numeric bytes, eight live bytes.
        without = fixture()
        previous = bind(without, ports=())
        with self.assertRaises(ValueError):
            bind(array, limits=NoiseLimits(previous.numeric_value_count, LIMITS.max_stored_numeric_bytes))
        with self.assertRaises(ValueError):
            bind(array, limits=NoiseLimits(LIMITS.max_sources, previous.retained_numeric_bytes + 15))
        layout = bind(array)
        self.assertEqual(layout.retained_numeric_bytes, previous.retained_numeric_bytes + 16)
        self.assertEqual(layout.sample_numeric_bytes, previous.sample_numeric_bytes + 8)
        before = array.volts, array.previous_field_volts
        original_work = PhysicalNoiseLayout.work
        def bad_work(bound, *args, **kwargs):
            return replace(original_work(bound, *args, **kwargs),
                           daq_external_source_interval_work_correction_joule=math.inf)
        with patch.object(PhysicalNoiseLayout, "work", bad_work):
            with self.assertRaisesRegex(ValueError, "source work must be finite"):
                advance(array, field_volts=np.zeros((64, 64)),
                        daq_selected_sources=selection(), noise_sample=sample(layout, .1))
        self.assertEqual((array.volts, array.previous_field_volts), before)
        plain, explicit = fixture(), fixture()
        kwargs = dict(field_volts=np.zeros((64, 64)), daq_selected_sources=selection())
        self.assertEqual(advance(plain, **kwargs), advance(explicit, noise_sample=None, **kwargs))
        self.assertIsNone(plain.physical_noise_layout)


if __name__ == "__main__":
    unittest.main()
