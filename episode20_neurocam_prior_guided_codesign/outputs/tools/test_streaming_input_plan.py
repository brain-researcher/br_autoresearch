"""Two adapter checks: invariant held inputs/charges and exact bounded coverage."""

import unittest

from array_acquisition import ProviderInterval
from raw_input_hold import MidpointHoldPlan
from streaming_acquisition_readout import ReadoutWindow
from streaming_input_plan import bind_hold_windows


def readout(index, start, end):
    return ReadoutWindow(index, start, end, (), (), (), (), ())


class StreamingInputPlanTests(unittest.TestCase):
    def plan(self):
        def provider(_):
            self.fail("Binding windows must never evaluate or resample the held field")
        return MidpointHoldPlan((
            ProviderInterval(0., 2., 5, 1., "reference", {"adc": 3}, ((0, 1e-12),)),
            ProviderInterval(2., 3., 7, 2.5, "reference", {"adc": None}, ((5, -2e-12),))),
            provider, (1e-8, 2e-8), {"max_intervals": 2})

    def test_seams_preserve_original_midpoints_event_ids_and_once_only_charge(self):
        hold = self.plan()
        windows = (readout(0, 0., 1.), readout(1, 1., 2.5), readout(2, 2.5, 3.))
        factory = bind_hold_windows(hold, lambda: iter(windows), max_windows=3,
                                   max_window_fragments=2, check=lambda: None)
        first, replay = tuple(factory()), tuple(factory())
        self.assertEqual(first, replay)
        fragments = [fragment for window in first for fragment in window.physical_intervals]
        self.assertEqual([fragment.event_index for fragment in fragments], [0, 0, 1, 1])
        self.assertEqual([(f.interval.start, f.interval.end) for f in fragments],
                         [(0., 1.), (1., 2.), (2., 2.5), (2.5, 3.)])
        self.assertEqual([f.interval.field_key for f in fragments], [1., 1., 2.5, 2.5])
        self.assertEqual([f.interval.injected_charge_coulomb for f in fragments],
                         [((0, 1e-12),), (), ((5, -2e-12),), ()])
        for fragment in fragments:
            original = hold.intervals[fragment.event_index]
            self.assertEqual(fragment.interval.gate_row, original.gate_row)
            self.assertEqual(fragment.interval.daq_selected_sources, original.daq_selected_sources)
            self.assertEqual(fragment.interval.voltage_reference_id, original.voltage_reference_id)
        # Coarser storage windows use identical input events, keys and charges.
        coarse = bind_hold_windows(hold, lambda: iter((readout(0, 0., 3.),)),
                                   max_windows=1, max_window_fragments=2, check=lambda: None)
        coarse_fragments = tuple(coarse())[0].physical_intervals
        self.assertEqual(tuple(fragment.interval for fragment in coarse_fragments), hold.intervals)
        self.assertEqual(hold.analytic_error_bounds_volts, (1e-8, 2e-8))

    def test_missing_overlap_extra_coverage_and_finite_window_fragment_caps_refuse(self):
        hold = self.plan()
        cases = (
            ((), 3, 2),
            ((readout(0, 0., 2.),), 3, 2),
            ((readout(0, 0., 1.), readout(1, 1.5, 3.)), 3, 2),
            ((readout(0, 0., 2.), readout(1, 1., 3.)), 3, 2),
            ((readout(0, 0., 4.),), 3, 2),
            ((readout(0, 0., 1.), readout(1, 1., 3.)), 1, 2),
            ((readout(0, 0., 3.),), 1, 1),
        )
        for windows, maximum, fragments in cases:
            with self.subTest(windows=windows, max_windows=maximum, max_fragments=fragments):
                factory = bind_hold_windows(hold, lambda: iter(windows), max_windows=maximum,
                                           max_window_fragments=fragments, check=lambda: None)
                with self.assertRaises(ValueError):
                    tuple(factory())


if __name__ == "__main__":
    unittest.main()
