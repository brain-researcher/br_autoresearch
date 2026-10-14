"""O1 planner pieces against the legacy hold planner and schedule (saved B2/2001).

Stub noise bounds use only IEEE basic operations and floor, so the legacy
scalar callback and the vectorized callback return identical floats. A
variable stub forces non-uniform refinement depth (1-2 levels per child).
"""
import json
import math
import unittest

import numpy as np

from conditional_acquisition_recipe import build_conditional_frontend, build_conditional_schedule
from conditional_forward_recipe import expected_state_ids
from development_simulation import load_saved_development_world
from fast_io_case import WORLD_PATH
from fast_io_exact import CYCLE_SECONDS, float_bits, shift
from fast_io_planner import FullRecordPlan, PlanningError, bisect_holds, template_from_schedule
from raw_world_time import RawWorldTime
from streaming_acquisition_readout import ReadoutWindow
from streaming_array_acquisition import _canonical
from streaming_input_plan import bind_hold_windows
from streaming_midpoint_hold import compile_streaming_midpoint_hold
from surface_to_contact import build_contact_factors, registered_pad_geometry


def variable_scalar(a, b, m):
    x = m * 1e4
    return (b - a) * (400000.0 + 1200000.0 * (x - math.floor(x)))


def variable_vector(a, b, m, slot_start, slot_end):
    x = m * 1e4
    return (b - a) * (400000.0 + 1200000.0 * (x - np.floor(x)))


def drifting_vector(a, b, m, slot_start, slot_end):
    # 3217*0.004 is not an integer, so the per-slot pattern changes between windows.
    x = m * 3217.0
    return (b - a) * (400000.0 + 1200000.0 * (x - np.floor(x)))


def constant_scalar(a, b, m):
    return 0.0187


def constant_vector(a, b, m, slot_start, slot_end):
    return np.full(np.shape(a), 0.0187)


def readout_bytes(window):
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    quantizers, chunks = {}, []
    def record(value):
        chunks.append(encoder.encode(_canonical(value, quantizers)).encode() + b"\n")
    record((window.index, window.start, window.end))
    for name in ("conversions", "returned_samples", "observations", "raw_last_use_windows",
                 "sample_last_use_windows", "nuisance_bindings"):
        sequence = getattr(window, name)
        record((name, len(sequence)))
        for item in sequence:
            record(item)
    return b"".join(chunks)


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check = lambda: None
        cls.world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
            max_elements=2000000, max_array_bytes=64 * 1024**2, max_metadata_bytes=1024**2, check=check)
        geometry = registered_pad_geometry("G0", short_axis="x", max_elements=2000000, check=check)
        cls.contacts = build_contact_factors(cls.world, geometry, max_elements=2000000, check=check)
        cls.raw_time = RawWorldTime(cls.world, max_elements=2000000, check=check)
        cls.frontend = build_conditional_frontend()
        cls.schedule = build_conditional_schedule(cls.frontend, state_ids=expected_state_ids(), check=check)
        cls.template = template_from_schedule(cls.schedule, cls.frontend)
        cls.bound = cls.raw_time.bind_contact_factors(cls.contacts)
        cls.lipschitz = cls.bound.derivative_bound_volts_per_second(0.0, 16.0)

    def legacy_plan(self, windows, scalar):
        end = float(shift(windows - 1, CYCLE_SECONDS))
        self.assertEqual(float_bits(self.bound.derivative_bound_volts_per_second(0.0, end)),
                         float_bits(self.lipschitz))
        return compile_streaming_midpoint_hold(self.raw_time, self.contacts,
            lambda: (a for k in range(windows) for a in self.schedule.physical_actions_for_cycle(k)),
            support=(0.0, end), epsilon_input_volts=1e-8, max_intervals=40000000, max_work=100000000,
            check=lambda: None, noise_hold_bound=scalar)

    def test_template_counts_and_digest_are_stable(self):
        t = self.template
        self.assertEqual((t.actions, t.conversions, t.returned, t.observations), (1600, 26000, 26000, 4496))
        self.assertEqual(template_from_schedule(self.schedule, self.frontend).digest, t.digest)
        self.assertEqual(t.input_ids, tuple(binding.input_id for binding in self.schedule.input_bindings))

    def test_variable_depth_bisection_matches_legacy(self):
        legacy = self.legacy_plan(2, variable_scalar)
        intervals = list(legacy.interval_factory())
        starts = shift(np.repeat([0, 1], 1600), np.tile(self.template.action_start, 2))
        ends = shift(np.repeat([0, 1], 1600), np.tile(self.template.action_end, 2))
        hs, he, hk, slot, field, noise, candidates = bisect_holds(
            starts, ends, lipschitz=self.lipschitz, epsilon=1e-8, noise_bound=variable_vector)
        self.assertEqual(len(intervals), len(hs))
        for name, values, new in (("start", [i.start for i in intervals], hs), ("end", [i.end for i in intervals], he),
                                  ("key", [i.field_key for i in intervals], hk)):
            self.assertTrue(np.array_equal(float_bits(np.array(values)), float_bits(new)), name)
        depth = np.bincount(slot, minlength=3200)
        self.assertGreater(len(set(depth.tolist())), 1, "stub must force non-uniform refinement")
        first = hs == starts[slot]
        self.assertEqual([bool(i.injected_charge_coulomb) for i in intervals],
                         (first & (np.diff(self.template.action_charge_ptr)[slot % 1600] > 0)).tolist())
        report = legacy.report
        self.assertEqual(report["candidate_bound_evaluations"], candidates)
        self.assertEqual(report["work_units"], 1 + 2 * candidates)
        self.assertEqual(float_bits(report["maximum_analytic_hold_bound_volts"]), float_bits(field.max()))
        self.assertEqual(float_bits(report["maximum_scaled_physical_noise_hold_bound"]), float_bits(noise.max()))

    def test_nonuniform_pattern_refuses_full_record_plan(self):
        # The cycle-periodic stub repeats window 0's pattern in every window and
        # is accepted; a stub that drifts between windows must refuse.
        periodic = FullRecordPlan(template=self.template, raw_time=self.raw_time, contact_factors=self.contacts,
                                  noise=None, noise_bound=variable_vector, epsilon_input_volts=1e-8,
                                  max_intervals=40000000, max_work=100000000, check=lambda: None)
        self.assertGreater(periodic.validate()["hold"]["compiled_intervals"], 12800000)
        plan = FullRecordPlan(template=self.template, raw_time=self.raw_time, contact_factors=self.contacts,
                              noise=None, noise_bound=drifting_vector, epsilon_input_volts=1e-8,
                              max_intervals=40000000, max_work=100000000, check=lambda: None)
        with self.assertRaisesRegex(PlanningError, "hold pattern"):
            plan.validate()

    def test_uniform_plan_windows_match_legacy_factory(self):
        plan = FullRecordPlan(template=self.template, raw_time=self.raw_time, contact_factors=self.contacts,
                              noise=None, noise_bound=constant_vector, epsilon_input_volts=1e-8,
                              max_intervals=40000000, max_work=100000000, check=lambda: None)
        totals = plan.validate()
        self.assertEqual((totals["hold"]["compiled_intervals"], totals["hold"]["candidate_bound_evaluations"]),
                         (12800000, 19200000))
        self.assertEqual(totals["split_intervals"], 25600000)
        for k in (0, 1, 3, 22, 38, 222, 384, 2224, 3846, 3999):
            legacy = readout_bytes(self.schedule.readout_window(k))
            self.assertEqual(readout_bytes(plan.legacy_readout_window(k)), legacy, k)
        legacy = self.legacy_plan(2, constant_scalar)
        windows = list(bind_hold_windows(legacy, lambda: (ReadoutWindow(k, float(shift(k, 0.0)),
            float(shift(k, CYCLE_SECONDS)), (), (), (), (), ()) for k in range(2)),
            max_windows=2, max_window_fragments=200000, check=lambda: None)())
        for k, window in enumerate(windows):
            mine = plan.legacy_window(k, readout_window=window.readout_window)
            self.assertEqual(len(mine.physical_intervals), len(window.physical_intervals))
            for a, b in zip(mine.physical_intervals, window.physical_intervals):
                self.assertEqual(json.dumps(_canonical(a, {}), sort_keys=True),
                                 json.dumps(_canonical(b, {}), sort_keys=True))
        self.assertEqual(legacy.report["compiled_intervals"], 6400)


if __name__ == "__main__":
    unittest.main()
