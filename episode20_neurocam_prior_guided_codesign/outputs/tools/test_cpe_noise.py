"""Three full-array physical noise seams; all numbers are algebra fixtures."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from physical_noise_ports import NoiseLimits, NodalCurrentPort, PhysicalNoiseLayout
from test_cpe_array import advance, fractional_contact, ideal_contact, make_array
from test_conditioned_cpe_array import (channel, fixture as conditioned_fixture,
                                       selections, solver_controls)


def bind(array, *, ports=(), amplifier=None, exclusions=()):
    return array.make_noise_layout(nodal_current_ports=ports, contact_noise_role="conditional_finite_band",
        amplifier_input_noise_role=amplifier, excluded_retained_contact_components=exclusions,
        limits=NoiseLimits(100000, 8 * 1024**2))


def point(layout, amplitudes=None):
    values = np.zeros(layout.numeric_value_count)
    for name, value in (amplitudes or {}).items():
        values[layout.port_ids.index(name)] = value
    return layout.sample_from_values(values)


class CPENoiseTests(unittest.TestCase):
    def test_full_fractional_contact_matches_independent_reduction_and_physical_port_work(self):
        contact = fractional_contact()
        h = np.array([contact.charge_transfer_siemens, *contact.branch_conductance_siemens])
        eta = np.array([.07, -.11, .13])
        for resistance in (.3, 0.):
            access = np.zeros((64, 64))
            access[0, 0] = resistance
            array = make_array(contacts=(contact,) + (ideal_contact(),) * 4095,
                access_resistance_ohm=access, tft_on_conductance_siemens=np.full((64, 64), .2),
                tft_off_conductance_siemens=np.full((64, 64), .2))
            spec = bind(array, ports=(NodalCurrentPort("source_to_pad", "source_00", "pad_00_00",
                                                       "conditional_current", "input_edge_00"),))
            e, supplied = (.2 if resistance else 0.), .17
            amplitudes = {f"contact:0000:star_branch:{i}": eta[i] for i in range(3)}
            amplitudes.update({"contact:0000:star_access_voltage": e, "source_to_pad": supplied})
            sample = point(spec, amplitudes)
            result = advance(array, noise_sample=sample)
            # Independent five-state reduction: exceptional pad, source0,
            # other63 equal pads, and the two retained CPE branch capacitors.
            c = np.diag([1., 1., 63. * 3., 0., 0.])
            for node, cap in zip((3, 4), contact.branch_capacitance_farad):
                edge = np.eye(5)[node] - np.eye(5)[0]
                c += cap * np.outer(edge, edge)
            g = np.zeros((5, 5))
            for first, second, conductance in ((0, 1, .2), (2, 1, 63. * .2)):
                edge = np.eye(5)[first] - np.eye(5)[second]
                g += conductance * np.outer(edge, edge)
            nodes, d = np.array([0, 3, 4]), 1 + resistance * h.sum()
            g[np.ix_(nodes, nodes)] += np.diag(h) - resistance * np.outer(h, h) / d
            current = np.zeros(5)
            current[nodes] = h / d + eta - resistance * h * eta.sum() / d + h * e / d
            current[1] += supplied
            current[0] -= supplied
            matrix = np.zeros((11, 11))
            matrix[:5, :5] = np.linalg.solve(c, -g)
            matrix[:5, 5] = np.linalg.solve(c, current)
            matrix[6:, :5] = np.eye(5)
            exact = expm(matrix * .02) @ np.r_[np.zeros(5), 1., np.zeros(5)]
            selected = [0, 4096, 64, 4160, 4161]
            np.testing.assert_allclose(np.array(result.kernel_interval.observation.end_volts)[selected],
                                       exact[:5], rtol=2e-10, atol=2e-12)
            mean = np.array(result.kernel_interval.observation.mean_volts)[selected]
            np.testing.assert_allclose(mean, exact[6:] / .02, rtol=2e-10, atol=2e-12)
            junction = (1 + e + resistance * (h @ mean[nodes] - eta.sum())) / d
            access_current = h @ (junction - mean[nodes]) + eta.sum()
            self.assertAlmostEqual(result.field_source_interval_work_joule, .02 * access_current, places=12)
            self.assertAlmostEqual(result.noise_access_voltage_source_interval_work_joule,
                                   .02 * e * access_current, places=12)
            self.assertAlmostEqual(result.noise_current_source_interval_work_joule,
                .02 * (eta @ (mean[nodes] - junction) + supplied * (mean[1] - mean[0])), places=12)
            self.assertEqual(result.field_source_boundary_work_joule, 0.)
            self.assertEqual(array.resources.total_states, 4162)
            self.assertEqual(array.resources.physical_noise_sources, spec.numeric_value_count)

    def test_amplifier_input_noise_uses_joint_preclip_drive_without_mutating_base(self):
        original = channel(gain=1.2, drive_lower_volts=-.25, drive_upper_volts=.25)
        array = conditioned_fixture(first=original, initial_source=.1)
        expected = conditioned_fixture(first=replace(original, offset_volts=.36), initial_source=.1)
        spec = bind(array, amplifier="conditional_input_referred_voltage")
        sample = point(spec, {"amplifier_input_voltage:00": .3})
        before_offsets = array._controlled_drives._offset.copy()
        kwargs = dict(controls=solver_controls(), daq_selected_sources=selections(), check=lambda: None)
        result = advance(array, noise_sample=sample, **kwargs)
        direct = advance(expected, **kwargs)
        np.testing.assert_allclose(array.volts, expected.volts, rtol=2e-11, atol=2e-12)
        self.assertAlmostEqual(result.controlled_voltage_source_interval_work_joule,
                               direct.controlled_voltage_source_interval_work_joule, places=13)
        self.assertAlmostEqual(result.conditioning_drive_branch_dissipation_joule,
                               direct.conditioning_drive_branch_dissipation_joule, places=13)
        self.assertEqual(result.noise_current_source_interval_work_joule, 0.)
        self.assertEqual(result.noise_access_voltage_source_interval_work_joule, 0.)
        np.testing.assert_array_equal(array._controlled_drives._offset, before_offsets)

    def test_retained_contact_ownership_invalid_samples_work_failure_and_none_compatibility(self):
        contact = replace(ideal_contact(), charge_transfer_siemens=.2)
        access = np.zeros((64, 64))
        access[0, 0] = .3
        array = make_array(contacts=(contact,) + (ideal_contact(),) * 4095, access_resistance_ohm=access)
        component = "contact:0000:charge_transfer"
        with self.assertRaisesRegex(ValueError, "retained charge-transfer"):
            bind(array)
        self.assertIsNone(array.physical_noise_layout)
        wrong = NodalCurrentPort("gct", "source_00", "pad_00_00", "passive_resistor_current", component)
        with self.assertRaisesRegex(ValueError, "retained charge-transfer"):
            bind(array, ports=(wrong,))
        port = NodalCurrentPort("gct", "contact_0000_interface", "pad_00_00",
                                "passive_resistor_current", component)
        spec = bind(array, ports=(port,))
        sample = point(spec, {"gct": .1, "contact:0000:star_branch:0": .2})
        before, field = array.volts, array.previous_field_volts
        with self.assertRaises(ValueError):
            advance(array)
        other = make_array()
        foreign = point(bind(other))
        with self.assertRaises(ValueError):
            advance(array, noise_sample=foreign)
        with patch.object(PhysicalNoiseLayout, "work", side_effect=ValueError("fixture accounting failure")):
            with self.assertRaises(ValueError):
                advance(array, noise_sample=sample)
        self.assertEqual((array.volts, array.previous_field_volts), (before, field))
        with self.assertRaises(ValueError):
            bind(array)
        omitted = make_array(contacts=(contact,) + (ideal_contact(),) * 4095, access_resistance_ohm=access)
        selected = bind(omitted, exclusions=((component, "explicit conditional omission; physical absence unknown"),))
        self.assertEqual(selected.excluded_retained_contact_components[0][0], component)
        plain, explicit = make_array(), make_array()
        first, second = advance(plain), advance(explicit, noise_sample=None)
        self.assertEqual(first, second)
        self.assertEqual((first.noise_current_source_interval_work_joule,
                          first.noise_access_voltage_source_interval_work_joule), (0., 0.))
        with self.assertRaises(ValueError):
            bind(plain)
        self.assertIsNone(plain.physical_noise_layout)


if __name__ == "__main__":
    unittest.main()
