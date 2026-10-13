"""Bounded planner regressions; synthetic bounds are numerical fixtures only."""
import unittest
from unittest.mock import Mock

import numpy as np

from raw_input_hold import PhysicalAction, HoldResolutionError, compile_midpoint_hold
from raw_world_time import RawWorldTime
from streaming_midpoint_hold import compile_streaming_midpoint_hold
from streaming_input_plan import bind_hold_windows
from streaming_acquisition_readout import ReadoutWindow


class StreamingMidpointHoldTests(unittest.TestCase):
    def setUp(self):
        self.raw = Mock(spec=RawWorldTime)
        self.raw.metadata = {"world_id": "bounded_fixture"}
        self.raw.breakpoints.return_value = np.array([0., .25, .5, 1.])
        self.bound = self.raw.bind_contact_factors.return_value
        self.bound.derivative_bound_volts_per_second.return_value = 1.
        self.actions = (
            PhysicalAction(0., .375, 0, "ref", {"adc": 0}, ((0, 1e-12),)),
            PhysicalAction(.375, 1., 1, "ref", {"adc": 1}, ((1, -1e-12),)),
        )
        self.limits = dict(epsilon_input_volts=.08, max_intervals=100, max_work=100,
                           check=lambda: None)

    def plan(self, factory=None, **overrides):
        limits = dict(self.limits, **overrides)
        return compile_streaming_midpoint_hold(self.raw, object(),
            factory or (lambda: iter(self.actions)), support=(0., 1.), **limits)

    def test_matches_retained_midpoints_and_once_only_charges_across_storage_seams(self):
        eager = compile_midpoint_hold(self.raw, object(), self.actions, **self.limits)
        streamed = self.plan()
        self.assertFalse(hasattr(streamed, "intervals"))
        self.assertEqual(tuple(streamed.interval_factory()), eager.intervals)
        self.assertEqual(tuple(streamed.interval_factory()), eager.intervals)
        self.assertEqual(streamed.report["compiled_intervals"], len(eager.intervals))
        self.assertEqual(streamed.report["maximum_analytic_hold_bound_volts"],
                         eager.report["maximum_analytic_hold_bound_volts"])
        def windows():
            yield ReadoutWindow(0, 0., .3, (), (), (), (), ())
            yield ReadoutWindow(1, .3, 1., (), (), (), (), ())
        args = dict(max_windows=2, max_window_fragments=100, check=lambda: None)
        self.assertEqual(tuple(bind_hold_windows(streamed, windows, **args)()),
                         tuple(bind_hold_windows(eager, windows, **args)()))
        self.bound.volts.assert_not_called()

    def test_late_schedule_failure_and_each_pass_limits_refuse(self):
        def gap():
            yield self.actions[0]
            yield PhysicalAction(.4, 1., 1, "ref", {"adc": 1})
        with self.assertRaises(ValueError):
            self.plan(gap)
        with self.assertRaises(ValueError):
            self.plan(lambda: iter(self.actions[:1]))
        with self.assertRaises(HoldResolutionError):
            self.plan(max_intervals=2)
        with self.assertRaises(HoldResolutionError):
            self.plan(max_work=1)
        holder = [self.actions]
        plan = self.plan(lambda: iter(holder[0]))
        holder[0] = self.actions[:1]
        with self.assertRaises(ValueError):
            tuple(plan.interval_factory())
        self.bound.volts.assert_not_called()

    def test_full_support_factory_is_consumed_without_retaining_action_or_hold_history(self):
        produced = []
        def factory():
            produced.append(0)
            for index in range(1024):
                produced[-1] += 1
                yield PhysicalAction(index / 1024, (index + 1) / 1024,
                                     index % 64, "ref", {"adc": index % 64})
        plan = self.plan(factory, max_intervals=1024, max_work=1100)
        self.assertEqual(produced, [1024])
        self.assertEqual(plan.report["compiled_intervals"], 1024)
        self.assertEqual(plan.report["retained_raw_boundary_count"], 4)
        self.assertFalse(plan.report["full_physical_action_or_hold_tuple_retained"])
        self.assertEqual(sum(1 for _ in plan.interval_factory()), 1024)
        self.assertEqual(produced, [1024, 1024])
        self.bound.volts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
