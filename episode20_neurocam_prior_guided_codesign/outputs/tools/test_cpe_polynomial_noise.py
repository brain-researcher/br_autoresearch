"""Full-array cubic forcing seams, with independent dense algebra fixtures."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from daq_mux_hold import (ExternalVoltageInput, PassiveDAQFrontend, PassiveMuxHoldGroup,
                          external_daq_branch_id)
from physical_noise_ports import (ExternalDAQCurrentPort, NoiseLimits, NodalCurrentPort,
                                  PhysicalNoiseLayout)
from polynomial_noise import NoisePolynomial
from test_conditioned_cpe_array import ADC, REF, channel, conditioning, selections, solver_controls
from test_cpe_array import advance, fractional_contact, ideal_contact, make_array


SELECTED = [0, 4096, 64, 4160, 4161, 4162, 4226, 4290]
WORK_FIELDS = ("noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule",
               "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
               "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule")
COEFFICIENTS = {
    "source_eta": (.07, 1.2, -3., 17.),
    "contact:0000:star_branch:0": (.02, -.4, 2., 50.),
    "contact:0000:star_branch:1": (-.03, .6, -5., 30.),
    "contact:0000:star_branch:2": (.01, -.8, 3., -70.),
    "contact:0000:star_access_voltage": (.08, 2., -4., 11.),
    "amplifier_input_voltage:00": (.03, 1., -7., 90.),
    "external_eta": (.04, -.7, 5., -20.),
}


def fixture(*, external_off=.02):
    membership = [(0,)] + [tuple(range(1 + 3*i, 4 + 3*i)) for i in range(17)]
    membership += [tuple(range(52 + 2*i, 54 + 2*i)) for i in range(6)]
    groups = tuple(PassiveMuxHoldGroup(f"conditioned_fixture_{index}", sources,
        .7, 0., (.8,) * len(sources), (.03,) * len(sources), .05,
        (ExternalVoltageInput("G1", .6, external_off),) if index == 0 else ())
        for index, sources in enumerate(membership))
    access = np.zeros((64, 64))
    access[0, 0] = .3
    array = make_array(contacts=(fractional_contact(),) + (ideal_contact(),) * 4095,
        access_resistance_ohm=access, analog_conditioning=conditioning(),
        daq_frontend=PassiveDAQFrontend(groups, "additional_fixed_shunt_excludes_frontend"),
        source_load_conductance_siemens=np.full(64, .03),
        tft_off_conductance_siemens=np.full((64, 64), .2))
    layout = array.make_noise_layout(
        nodal_current_ports=(NodalCurrentPort("source_eta", "source_00", None,
            "conditional_finite_band", "source_current_fixture"),),
        contact_noise_role="conditional_finite_band", amplifier_input_noise_role="conditional_input_voltage",
        excluded_retained_contact_components=(),
        external_daq_current_ports=(ExternalDAQCurrentPort("external_eta", external_daq_branch_id(ADC, "G1"),
            ADC, "G1", "daq_hold_" + ADC, "conditional_finite_band"),),
        limits=NoiseLimits(100000, 8 * 1024**2))
    return array, layout


def polynomial(layout, amplitudes=COEFFICIENTS, *, origin=1.01, start=1., end=1.02):
    rows = np.zeros((4, layout.numeric_value_count))
    for name, coefficients in amplitudes.items():
        rows[:, layout.port_ids.index(name)] = coefficients
    return NoisePolynomial(layout, rows, origin, start, end)


def run(array, noise, *, start=1., duration=.02, charge=None, **changes):
    controls = solver_controls(relative_tolerance=2e-6, voltage_atol=1e-9,
        charge_atol_coulomb=1e-10, energy_atol_joule=1e-9,
        max_operator_products=1000000, max_linear_iterations=100000)
    kwargs = dict(noise_sample=noise, interval_start_seconds=start, duration_seconds=duration,
        controls=controls, check=lambda: None, daq_selected_sources=selections("G1", .2))
    if charge is not None:
        kwargs["injected_charge_coulomb"] = charge
    kwargs.update(changes)
    return advance(array, **kwargs)


def reference(duration=.02, origin=1.01, start=1.):
    """Eight-state physical reduction plus polynomial clocks and exact moments.

    Coordinates are exceptional pad/source/other63 pads/two contact states/
    buffer/filter/hold. Original contact junction/current expressions construct
    the work integrals independently of the production projection and solver.
    """
    contact, spec = fractional_contact(), channel()
    h = np.r_[contact.charge_transfer_siemens, contact.branch_conductance_siemens]
    nodes, resistance = np.array([0, 3, 4]), .3
    d = 1 + resistance * h.sum()
    c = np.diag([1., 1.5, 189., 0., 0., .8, .7, .7])
    for node, cap in zip((3, 4), contact.branch_capacitance_farad):
        edge = np.eye(8)[node] - np.eye(8)[0]
        c += cap * np.outer(edge, edge)
    g = np.diag([0., .08, 0., 0., 0., .9, .04, .65])
    for first, second, conductance in ((0, 1, .2), (2, 1, 12.6), (5, 6, .6), (6, 7, .03)):
        edge = np.eye(8)[first] - np.eye(8)[second]
        g += conductance * np.outer(edge, edge)
    g[np.ix_(nodes, nodes)] += np.diag(h) - resistance * np.outer(h, h) / d
    eta = np.array([COEFFICIENTS[f"contact:0000:star_branch:{i}"] for i in range(3)])
    access = np.array(COEFFICIENTS["contact:0000:star_access_voltage"])
    source = np.array(COEFFICIENTS["source_eta"])
    amplifier = np.array(COEFFICIENTS["amplifier_input_voltage:00"])
    external = np.array(COEFFICIENTS["external_eta"])
    force = np.zeros((8, 4))
    force[nodes] = eta + h[:, None] * (access - resistance * eta.sum(axis=0)) / d
    force[nodes, 0] += h / d
    force[1] += source
    force[5] += .9 * 1.2 * amplifier
    force[7] += external
    force[7, 0] += .6 * .2
    matrix = -g
    matrix[5, 1] += .9 * 1.2
    generator = np.zeros((12, 12))
    generator[:8, :8] = np.linalg.solve(c, matrix)
    generator[:8, 8:] = np.linalg.solve(c, force)
    generator[9, 8], generator[10, 9], generator[11, 10] = 1., 2., 3.
    delta = start - origin
    initial = np.r_[np.zeros(8), 1., delta, delta**2, delta**3]
    augmented = np.zeros((20, 20))
    augmented[:12, :12] = generator
    augmented[12:, :8] = np.eye(8)
    values = expm(duration * augmented) @ np.r_[initial, np.zeros(8)]
    basis = np.eye(12)
    eta_forms = np.column_stack((np.zeros((3, 8)), eta))
    e, n, amp, ext_eta = (np.r_[np.zeros(8), v] for v in (access, source, amplifier, external))
    field, ext = basis[8], .2 * basis[8]
    junction = (field + e + resistance * (h @ basis[nodes] - eta_forms.sum(axis=0))) / d
    access_current = h.sum() * junction - h @ basis[nodes] + eta_forms.sum(axis=0)
    u = 1.2 * (basis[1] + amp)
    current_work = np.outer(n, basis[1]) + np.outer(ext_eta, basis[7] - ext)
    for noise, node in zip(eta_forms, nodes):
        current_work += np.outer(noise, basis[node] - junction)
    forms = (current_work, np.outer(e, access_current), np.outer(field, access_current),
             .6 * np.outer(ext, ext - basis[7]) + np.outer(ext, ext_eta),
             .9 * np.outer(u, u - basis[5]), .9 * np.outer(u - basis[5], u - basis[5]))
    moments = np.zeros((150, 150))
    moments[:144, :144] = np.kron(generator, np.eye(12)) + np.kron(np.eye(12), generator)
    moments[144:, :144] = np.array([form.ravel() for form in forms])
    work = (expm(duration * moments) @ np.r_[np.kron(initial, initial), np.zeros(6)])[144:]
    return values[:8], values[12:] / duration, work


class CPEPolynomialNoiseTests(unittest.TestCase):
    def test_full_array_cubic_endpoint_means_and_original_physical_work(self):
        array, layout = fixture()
        result = run(array, polynomial(layout))
        expected, mean, work = reference()
        observation = result.kernel_interval.observation
        np.testing.assert_allclose(np.asarray(observation.end_volts)[SELECTED], expected,
                                   rtol=1e-5, atol=3e-8)
        np.testing.assert_allclose(np.asarray(observation.mean_volts)[SELECTED], mean,
                                   rtol=1e-5, atol=3e-8)
        np.testing.assert_allclose([getattr(result, name) for name in WORK_FIELDS], work,
                                   rtol=2e-5, atol=3e-9)
        self.assertEqual(array.resources.total_states, 4314)
        self.assertEqual(len(result.daq_hold_observations), 24)
        self.assertNotEqual(result.kernel_interval.time_varying_current_work_joule, 0.)
        self.assertEqual(len(result.kernel_interval.physical_noise_work_joule), 4)

    def test_same_cubic_crosses_seams_with_charge_once_and_no_mean_product(self):
        whole, first_layout = fixture()
        split, second_layout = fixture()
        first_noise, second_noise = polynomial(first_layout), polynomial(second_layout)
        frozen = second_noise.numeric_array.copy()
        charge = np.zeros(len(whole.state_ids))
        charge[4096] = .03
        full = run(whole, first_noise, charge=charge)
        left = run(split, second_noise, duration=.007, charge=charge)
        right = run(split, second_noise, start=1.007, duration=.013)
        np.testing.assert_allclose(split.volts, whole.volts, rtol=2e-5, atol=5e-8)
        combined_mean = (.007 * np.asarray(left.kernel_interval.observation.mean_volts)
                         + .013 * np.asarray(right.kernel_interval.observation.mean_volts)) / .02
        np.testing.assert_allclose(combined_mean, full.kernel_interval.observation.mean_volts,
                                   rtol=2e-5, atol=5e-8)
        for name in WORK_FIELDS:
            self.assertAlmostEqual(getattr(full, name), getattr(left, name) + getattr(right, name), delta=5e-9)
        self.assertEqual(right.supplied_charge_boundary_work_joule, 0.)
        np.testing.assert_array_equal(second_noise.numeric_array, frozen)
        held, held_layout = fixture()
        held_result = run(held, polynomial(held_layout).sample_at(1.01), interval_start_seconds=None)
        exact, _, _ = reference()
        self.assertGreater(np.max(np.abs(np.asarray(held_result.kernel_interval.observation.end_volts)[SELECTED]
                                         - exact)), 1e-7)

    def test_support_late_zero_branch_and_accounting_refusals_are_atomic(self):
        array, layout = fixture(external_off=0.)
        # Only the cubic coefficient is nonzero: midpoint-only checking would
        # miss this forbidden current on an unselected zero-conductance path.
        noise = polynomial(layout, {"external_eta": (0., 0., 0., 1.)})
        before, field = array.volts, array.previous_field_volts
        off = selections(None, .2)
        with self.assertRaises(ValueError):
            array.validate_noise_interval(noise, daq_selected_sources=off)
        with self.assertRaises(ValueError):
            run(array, noise, daq_selected_sources=off)
        valid = polynomial(layout, {})
        for changes in ({"start": None}, {"start": .999}, {"duration": .021}):
            with self.assertRaises(ValueError):
                run(array, valid, **changes)
        with patch.object(PhysicalNoiseLayout, "work", side_effect=ValueError("late physical work refusal")):
            with self.assertRaisesRegex(ValueError, "late physical work"):
                run(array, valid)
        self.assertEqual((array.volts, array.previous_field_volts), (before, field))
        old = array._assembly_limits
        array._assembly_limits = replace(old, max_stored_numeric_bytes=array.resources.stored_numeric_bytes)
        with self.assertRaisesRegex(ValueError, "numeric-storage"):
            run(array, valid)
        self.assertEqual((array.volts, array.previous_field_volts), (before, field))


if __name__ == "__main__":
    unittest.main()
