"""Three deterministic physical-port identities; no Monte Carlo or PSD fit."""
from dataclasses import FrozenInstanceError
from itertools import repeat
import unittest

import numpy as np

from physical_noise_ports import NoiseLimits, NodalCurrentPort, PhysicalNoiseLayout


def layout(*, resistance=3., limits=None):
    return PhysicalNoiseLayout(state_ids=("v0", "v1", "v2", "v3"), voltage_reference_id="common",
        nodal_current_ports=(NodalCurrentPort("edge", "v2", "v3", "conditional_current", "edge_23"),
                             NodalCurrentPort("shunt", "v0", None, "conditional_current", "input_0")),
        contact_stars=(((0, 1, 2), (.2, .7, 1.1), resistance), ((3,), (.4,), 0.))
                      + (((3,), (0.,), 0.),) * 4094,
        contact_ownership="all_eliminated_star_branch_and_access_noise_owned_here",
        contact_noise_role="explicit_finite_band_branch_sources", amplifier_input_noise_role=None,
        excluded_retained_contact_components=(), limits=limits or NoiseLimits(20000, 2**20))


def sample(spec, **amplitudes):
    values = np.zeros(spec.numeric_value_count)
    for name, value in amplitudes.items():
        values[spec.port_ids.index(name)] = value
    return spec.sample_from_values(values)


class PhysicalNoisePortsTests(unittest.TestCase):
    def test_eliminated_equilibrium_covariance_matches_full_star_with_negative_cross_terms(self):
        h = np.array([.2, .7, 1.1])
        for resistance in (3., 0.):
            spec = layout(resistance=resistance)
            columns = []
            # Unit 4*kB*T factor: physical branch PSDs h, access-voltage PSD R.
            for branch, amplitude in enumerate(np.sqrt(h)):
                point = sample(spec, **{f"contact:0000:star_branch:{branch}": amplitude})
                columns.append(spec.project_current(point)[:3])
            point = sample(spec, **{"contact:0000:star_access_voltage": np.sqrt(resistance)})
            columns.append(spec.project_current(point)[:3])
            factor = np.column_stack(columns)
            expected = np.diag(h) - resistance * np.outer(h, h) / (1 + resistance * h.sum())
            np.testing.assert_allclose(factor @ factor.T, expected, rtol=2e-14, atol=2e-15)
            if resistance:
                self.assertLess((factor @ factor.T)[0, 1], 0.)
            ideal = sample(spec, **{"contact:0001:star_branch:0": np.sqrt(.4)})
            self.assertAlmostEqual(spec.project_current(ideal)[3] ** 2, .4)

    def test_original_junction_power_and_zero_access_port_cancellation(self):
        h, eta = np.array([.2, .7, 1.1]), np.array([.07, -.11, .13])
        mean = np.array([.3, -.1, .6, .4])
        duration, voltage, ideal_voltage, ideal_eta = .2, 1.2, -.4, .3
        for resistance in (3., 0.):
            spec = layout(resistance=resistance)
            access = .2 if resistance else 0.
            amplitudes = {f"contact:0000:star_branch:{i}": eta[i] for i in range(3)}
            amplitudes.update({"contact:0000:star_access_voltage": access,
                "contact:0001:star_branch:0": ideal_eta, "edge": -.2, "shunt": .17})
            point = sample(spec, **amplitudes)
            field = np.zeros(4096)
            field[:2] = voltage, ideal_voltage
            result = spec.work(point, mean_state_volts=mean, field_port_volts=field, duration_seconds=duration)
            d = 1 + resistance * h.sum()
            junction = (voltage + access + resistance * (h @ mean[:3] - eta.sum())) / d
            access_current = h @ (junction - mean[:3]) + eta.sum()
            ideal_current = .4 * (ideal_voltage - mean[3]) + ideal_eta
            ordinary_power = -.2 * (mean[2] - mean[3]) + .17 * mean[0]
            expected_current_power = eta @ (mean[:3] - junction) + ideal_eta * (mean[3] - ideal_voltage) + ordinary_power
            baseline_field_power = voltage * (h.sum() * voltage - h @ mean[:3]) / d
            baseline_field_power += ideal_voltage * .4 * (ideal_voltage - mean[3])
            full_field_power = voltage * access_current + ideal_voltage * ideal_current
            self.assertAlmostEqual(result.noise_current_source_interval_work_joule, duration * expected_current_power)
            self.assertAlmostEqual(result.noise_access_voltage_source_interval_work_joule, duration * access * access_current)
            self.assertAlmostEqual(result.field_source_interval_work_correction_joule,
                                   duration * (full_field_power - baseline_field_power))
            heat = resistance * access_current**2 + h @ ((junction - mean[:3])**2)
            heat += .4 * (ideal_voltage - mean[3])**2
            physical_power = full_field_power + access * access_current + expected_current_power - heat
            retained_current = np.r_[h * (junction - mean[:3]) + eta, ideal_current]
            retained_current += np.array([.17, 0., -.2, .2])
            self.assertAlmostEqual(physical_power, mean @ retained_current, places=14)
            if not resistance:
                # Field correction cancels the −U*sum(eta) part of source work.
                correction = result.field_source_interval_work_correction_joule
                noise_work = result.noise_current_source_interval_work_joule
                self.assertAlmostEqual(correction + noise_work,
                    duration * (eta @ mean[:3] + ideal_eta * mean[3] + ordinary_power), places=14)

    def test_owned_original_coordinates_limits_and_ideal_port_refusals(self):
        spec = layout()
        values = np.zeros(spec.numeric_value_count)
        point = spec.sample_from_values(values)
        values[0] = 9.
        self.assertEqual(point.numeric_values[0], 0.)
        np.testing.assert_array_equal(point.numeric_array, np.zeros(spec.numeric_value_count))
        with self.assertRaises(ValueError):
            point.numeric_array.flags.writeable = True
        with self.assertRaises(ValueError):
            point.numeric_array[0] = 1.
        for invalid in (np.zeros(spec.numeric_value_count, dtype=bool),
                        np.zeros(spec.numeric_value_count, dtype=complex),
                        np.zeros((spec.numeric_value_count, 1))):
            with self.assertRaises(ValueError):
                spec.sample_from_values(invalid)
        supplied = np.zeros(2 * spec.numeric_value_count)[::2]
        frozen = spec.sample_from_values(supplied)
        supplied[0] = 5.
        self.assertEqual(frozen.numeric_values, point.numeric_values)
        self.assertEqual(point.state_ids, spec.state_ids)
        self.assertEqual(point.voltage_reference_id, "common")
        self.assertEqual(spec.sample_numeric_bytes, 8 * spec.numeric_value_count)
        self.assertEqual(spec.port_units[:2], ("A", "A"))
        self.assertEqual(spec.port_units[-4096:], ("V",) * 4096)
        with self.assertRaises(FrozenInstanceError):
            point.numeric_values = ()
        for name in ("contact:0001:star_access_voltage", "contact:0002:star_branch:0"):
            with self.subTest(port=name), self.assertRaises(ValueError):
                sample(spec, **{name: 1.})
        with self.assertRaises(ValueError):
            spec.sample_from_values(np.zeros(spec.numeric_value_count - 1))
        with self.assertRaises(ValueError):
            sample(spec, edge=float("nan"))
        with self.assertRaises(ValueError):
            spec.project_current(sample(layout()))
        for mean, field in ((np.zeros(4, dtype=complex), np.zeros(4096)),
                            (np.zeros(4), np.zeros(4096, dtype=complex))):
            with self.assertRaisesRegex(ValueError, "real voltages"):
                spec.work(point, mean_state_volts=mean, field_port_volts=field, duration_seconds=.2)
        for limits in (NoiseLimits(1, 2**20), NoiseLimits(20000, 1)):
            with self.assertRaises(ValueError):
                layout(limits=limits)
        arguments = {key: getattr(spec, key) for key in ("state_ids", "voltage_reference_id",
            "nodal_current_ports", "contact_stars", "contact_ownership", "contact_noise_role",
            "amplifier_input_noise_role", "excluded_retained_contact_components", "limits")}
        with self.assertRaisesRegex(ValueError, "finite item limit"):
            PhysicalNoiseLayout(**{**arguments, "nodal_current_ports": repeat(spec.nodal_current_ports[0]),
                                   "limits": NoiseLimits(2, 2**20)})
        with self.assertRaisesRegex(ValueError, "finite item limit"):
            PhysicalNoiseLayout(**{**arguments, "state_ids": repeat("v0"),
                                   "limits": NoiseLimits(20000, 8)})
        duplicated = NodalCurrentPort("extra", "v0", None, "conditional_current",
                                     "contact:0000:star_branch:0")
        with self.assertRaisesRegex(ValueError, "cannot also have nodal owners"):
            PhysicalNoiseLayout(**{**arguments, "nodal_current_ports": (duplicated,)})


if __name__ == "__main__":
    unittest.main()
