"""Six bounded analytic-hold checks; private fixtures, no scientific payloads."""
from fractions import Fraction
from unittest.mock import patch
import unittest

import numpy as np

from development_worlds import CONDITIONS, _Grid, _build
from raw_world_time import (RawWorldTime, PI_UPPER, _BoundContactTime,
                            _upper_add, _upper_mul, _upper_div)
from surface_to_contact import ContactFactors
from raw_input_hold import PhysicalAction, HoldResolutionError, compile_midpoint_hold

LIMIT = 20_000_000


class RawInputHoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worlds = {condition: _build(condition, 2001, _Grid(48, 48),
            check=lambda: None, max_elements=LIMIT, include_scales=False) for condition in CONDITIONS}
        cls.models = {condition: RawWorldTime(world, max_elements=LIMIT)
                      for condition, world in cls.worlds.items()}

    def contact(self, condition, *, zero=False):
        model, world = self.models[condition], self.worlds[condition]
        matrix = np.zeros((4096, len(world.raw_drivers)))
        if not zero:
            matrix[:, 0] = 1.
        return ContactFactors(matrix, world.raw_drivers.copy(), {
            "world_id": model.metadata["world_id"], "independence_group": model.metadata["independence_group"],
            "role": "development_preparation_unassigned"})

    def test_outward_nonnegative_arithmetic_against_exact_rationals_and_contact_sum(self):
        cases = [(0., 0.), (0., 1.), (.1, .2), (1., 2**-53),
                 (np.nextafter(0., 1.), .5), (1e-200, 1e-110), (1e200, 1e-200)]
        for a, b in cases:
            a, b = float(a), float(b)
            exact_a, exact_b = Fraction.from_float(a), Fraction.from_float(b)
            self.assertGreaterEqual(Fraction.from_float(_upper_add(a, b)), exact_a + exact_b)
            self.assertGreaterEqual(Fraction.from_float(_upper_mul(a, b)), exact_a * exact_b)
            if b:
                exact_quotient = exact_a / exact_b
                if exact_quotient > Fraction.from_float(float.fromhex("0x1.fffffffffffffp+1023")):
                    with self.assertRaises(ValueError):
                        _upper_div(a, b)
                else:
                    self.assertGreaterEqual(Fraction.from_float(_upper_div(a, b)), exact_quotient)
        self.assertGreater(Fraction.from_float(PI_UPPER), Fraction("3.1415926535897932384626433832795028841971693993751"))
        with self.assertRaises(ValueError):
            _upper_mul(1e308, 1e308)
        contact = self.contact("B2")
        contact.contact_modes[0] = [1., -.2]
        bound = self.models["B2"].bind_contact_factors(contact)
        rates = self.models["B2"].derivative_bounds(0., .1)
        exact = max(sum(Fraction.from_float(abs(float(a))) * Fraction.from_float(float(rate))
                        for a, rate in zip(row, rates)) for row in contact.contact_modes) / 1_000_000
        self.assertGreaterEqual(Fraction.from_float(bound.derivative_bound_volts_per_second(0., .1)), exact)

    def test_all28_driver_enclosures_and_zero_prelude_without_dense_sampling(self):
        for condition, model in self.models.items():
            edges = model.breakpoints()
            selected = None
            for a, b in zip(edges, edges[1:]):
                rates = model.derivative_bounds(float(a), float(b))
                if np.any(rates):
                    selected = float(a), float(b), rates
                    break
            self.assertIsNotNone(selected, condition)
            a, b, rates = selected
            points = a + (b - a) * np.array([.25, .5, .75])
            step = min(1e-7, (b - a) * 1e-5)
            observed = (model.drivers_uv(points + step) - model.drivers_uv(points - step)) / (2 * step)
            # Numerical derivative agreement is a regression check; the analytic
            # inequalities in derivative_bounds establish the interval bound.
            self.assertTrue(np.all(np.abs(observed) <= rates[:, None] * (1 + 1e-6) + 1e-6), condition)
        np.testing.assert_array_equal(self.models["S2"].derivative_bounds(0., .1), 0.)
        with self.assertRaises(ValueError):
            self.models["M4"].derivative_bounds(.9, 1.1)
        first = self.worlds["S2"].metadata["details"]["events"][0]["support_start_s"]
        with self.assertRaises(ValueError):
            self.models["S2"].derivative_bounds(first - .001, first + .001)

    def test_midpoint_sine_error_against_exact_extrema_and_deterministic_bisection(self):
        model, contact = self.models["B4"], self.contact("B4")
        actions = [PhysicalAction(0., .02, 0, "fixture_reference", None)]
        kwargs = dict(epsilon_input_volts=1e-6, max_intervals=4096, max_work=8192)
        plan = compile_midpoint_hold(model, contact, actions, **kwargs)
        repeat = compile_midpoint_hold(model, contact, actions, **kwargs)
        self.assertEqual(plan.intervals, repeat.intervals)
        self.assertEqual(plan.analytic_error_bounds_volts, repeat.analytic_error_bounds_volts)
        self.assertGreater(len(plan.intervals), 1)
        scale = self.worlds["B4"].metadata["details"]["applied_scale"]
        omega = 2 * np.pi * 80.
        for interval, error in zip(plan.intervals, plan.analytic_error_bounds_volts):
            self.assertLessEqual(error, 1e-6)
            self.assertLess(interval.start, interval.field_key)
            self.assertLess(interval.field_key, interval.end)
            extrema = [interval.start, interval.end]
            extrema += [k * np.pi / omega for k in range(-1, 20)
                        if interval.start < k * np.pi / omega < interval.end]
            actual_max = max(abs(scale * (np.cos(omega * t) - np.cos(omega * interval.field_key)))
                             for t in extrema) / 1e6
            self.assertLessEqual(actual_max, error + 1e-17)
            expected = scale * np.cos(omega * interval.field_key) / 1e6
            np.testing.assert_allclose(plan.field_provider(interval.field_key), expected, rtol=2e-14, atol=1e-17)
        self.assertFalse(plan.report["waveform_evaluation_arithmetic_error_included"])
        self.assertFalse(plan.report["circuit_output_error_included"])

    def test_required_boundaries_M4_jumps_and_parent_charge_once_only(self):
        model, contact = self.models["S2"], self.contact("S2", zero=True)
        actions = [PhysicalAction(0., 1., 0, "fixture_reference", {"adc": None}, ((0, .1),)),
                   PhysicalAction(1., 16., 3, "fixture_reference", {"adc": 4}, ((7, -.2),))]
        extras = [.3, 1.2, 3.3]
        plan = compile_midpoint_hold(model, contact, actions, epsilon_input_volts=1e-6,
            max_intervals=1024, max_work=4096, additional_boundaries_s=extras)
        starts = {interval.start for interval in plan.intervals}
        for point in list(model.breakpoints()[:-1]) + extras:
            self.assertIn(point, starts)
        charged = [event for event in plan.intervals if event.injected_charge_coulomb]
        self.assertEqual([(event.start, event.injected_charge_coulomb) for event in charged],
                         [(0., ((0, .1),)), (1., ((7, -.2),))])
        for event in plan.intervals:
            self.assertEqual(event.gate_row, 0 if event.start < 1. else 3)
            self.assertEqual(dict(event.daq_selected_sources), {"adc": None if event.start < 1. else 4})
        self.assertTrue(all(error == 0. for error in plan.analytic_error_bounds_volts))
        m4 = compile_midpoint_hold(self.models["M4"], self.contact("M4"),
            [PhysicalAction(.9, 1.1, 0, "fixture_reference", None)],
            epsilon_input_volts=1., max_intervals=8, max_work=16)
        self.assertEqual([(event.start, event.end) for event in m4.intervals], [(.9, 1.), (1., 1.1)])
        self.assertFalse(any(event.start < 1. < event.end for event in m4.intervals))

    def test_explicit_work_interval_precision_and_schedule_refusals(self):
        model, contact = self.models["B4"], self.contact("B4")
        action = PhysicalAction(.001, .002, 0, "fixture_reference", None)
        for kwargs in ({"epsilon_input_volts": 1e-12, "max_intervals": 1, "max_work": 100},
                       {"epsilon_input_volts": 1., "max_intervals": 100, "max_work": 1},
                       {"epsilon_input_volts": float(np.nextafter(0., 1.)), "max_intervals": 1024, "max_work": 4096}):
            with self.assertRaises(HoldResolutionError):
                compile_midpoint_hold(model, contact, [action], **kwargs)
        with self.assertRaises(ValueError):
            compile_midpoint_hold(model, contact, [action], epsilon_input_volts=0., max_intervals=10, max_work=20)
        with self.assertRaises(ValueError):
            compile_midpoint_hold(model, contact, [action], epsilon_input_volts=1., max_intervals=10,
                                  max_work=20, additional_boundaries_s=[.003])
        with self.assertRaises(ValueError):
            compile_midpoint_hold(model, contact, [action, PhysicalAction(.003, .004, 0, "fixture_reference", None)],
                                  epsilon_input_volts=1., max_intervals=10, max_work=20)
        with self.assertRaises(ValueError):
            PhysicalAction(0., .1, 64, "fixture_reference", None)

    def test_cached_bounds_no_voltage_plan_and_owned_provider_after_caller_mutation(self):
        model, contact = self.models["B2"], self.contact("B2", zero=True)
        selection = {"adc": None}
        actions = [PhysicalAction(index / 1000., (index + 1) / 1000., index % 64,
                    "fixture_reference", selection) for index in range(100)]
        with patch.object(_BoundContactTime, "volts", side_effect=AssertionError("no field plan may be generated")):
            plan = compile_midpoint_hold(model, contact, actions, epsilon_input_volts=1e-6,
                                         max_intervals=100, max_work=101)
        self.assertEqual(plan.report["cached_raw_segment_derivative_enclosures"], 1)
        self.assertEqual(plan.report["work_units"], 101)
        self.assertEqual(len(plan.intervals), 100)
        self.assertTrue(all(not hasattr(event, "field_volts") for event in plan.intervals))
        self.assertFalse(plan.report["dense_voltage_plan_materialized"])
        self.assertFalse(plan.report["acquisition_policy_receives_future_fields_or_splits"])
        contact.contact_modes[:] = 100.
        contact.raw_drivers[:] = 99.
        selection["adc"] = 63
        actions.clear()
        np.testing.assert_array_equal(plan.field_provider(plan.intervals[0].field_key), 0.)
        self.assertIsNone(plan.intervals[0].daq_selected_sources["adc"])
        with self.assertRaises(TypeError):
            plan.report["epsilon_input_volts"] = 100.


if __name__ == "__main__":
    unittest.main()
