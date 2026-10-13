"""One guide/axis/role restriction check; coordinated Slurm execution only."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bandwise_prediction import BandwiseLFP, BANDS
from component_scoring import BoundActivity, FrozenDesign
from development_views import DevelopmentRole
import electrode_unit_intersection as core


def check():
    pass


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("neural values were read before paired metadata binding")


def metadata(region, guide, positions):
    guide = tuple(tuple(float(value) for value in row) for row in guide)
    axes = tuple(f"{region}:electrode_{electrode!r}:unit_{unit!r}" for electrode, unit in guide)
    directions = tuple(index // 2 for index in range(16))
    design = FrozenDesign("Mihili_CO_20140217.mat", "training", tuple(range(8)),
        tuple(f"training_{index}" for index in range(16)), directions, tuple(range(5, 15)), axes)
    electrodes = tuple(float(value) for value in range(1, 97))
    return DevelopmentRole("Mihili", region, design, tuple(range(16)),
        tuple(value * math.pi / 4 for value in directions), (-1,) * 16,
        tuple(range(16)), tuple(value + 10 for value in range(16)), electrodes,
        positions, tuple(electrodes[index] for index in positions), guide)


class ElectrodeUnitIntersectionTests(unittest.TestCase):
    def test_exact_guide_membership_preserves_units_trials_features_and_refuses_unmatched_support_before_values(self):
        left = metadata("M1", ((20, 7), (3, 0), (1, 5), (3, 2), (40, 1)), tuple(range(15)))
        right = metadata("PMd", ((45, 1), (2, 7), (33, 2), (33, 0), (64, 1), (44, 9)), tuple(range(30, 45)))
        plan = core.plan_paired_unit_intersection(left, right, check=check)
        self.assertEqual(plan.required_population_rank, 2)
        self.assertEqual(plan.m1.retained_unit_indices, (1, 2, 3))
        self.assertEqual(plan.pmd.retained_unit_indices, (0, 2, 3, 5))
        self.assertEqual(plan.m1.retained_unit_guide, ((3., 0.), (1., 5.), (3., 2.)))
        self.assertEqual(plan.pmd.retained_unit_guide, ((45., 1.), (33., 2.), (33., 0.), (44., 9.)))
        self.assertEqual(plan.m1.selected_electrodes_with_units, (1., 3.))
        self.assertEqual(plan.pmd.selected_electrodes_with_units, (33., 44., 45.))
        # Distinct native unit counts and physical electrode sets must remain
        # distinct. Even unreadable LFP payloads need no array access to bind
        # the guide-selected count axes for subsequent fitting/prediction.
        lfp = tuple(BandwiseLFP(meta.region, meta.design, meta.selected_electrode_ids,
                               BANDS, ForbiddenArray()) for meta in (left, right))
        features = core.restrict_paired_features(*lfp, plan=plan, check=check)
        for original, selected, regional in zip(lfp, features, (plan.m1, plan.pmd)):
            self.assertIs(selected.values, original.values)
            self.assertEqual(selected.electrode_ids, original.electrode_ids)
            self.assertEqual(selected.bands, original.bands)
            self.assertEqual(selected.design.axis_ids, regional.retained_axis_ids)
            self.assertEqual(replace(selected.design, axis_ids=original.design.axis_ids), original.design)
        arrays = tuple(np.arange(math.prod(meta.design.shape), dtype=float).reshape(meta.design.shape)
                       for meta in (left, right))
        activity = tuple(BoundActivity(meta.design, values) for meta, values in zip((left, right), arrays))
        selected = core.restrict_paired_activity(*activity, plan=plan, check=check)
        for index, (before, after, indices) in enumerate(zip(activity, selected, ((1, 2, 3), (0, 2, 3, 5)))):
            np.testing.assert_array_equal(after.values, arrays[index][:, :, indices])
            self.assertEqual(after.design, features[index].design)
            self.assertEqual(after.values.shape[:2], before.values.shape[:2])
            with self.assertRaises(ValueError):
                after.values.setflags(write=True)
        # Zero-rank evaluation counts remain present: inclusion is metadata
        # only, and only the caller's TRAINING fit may assess numerical rank2.
        evaluation = tuple(replace(meta.design, role="evaluation",
            trial_ids=tuple(f"evaluation_{index}" for index in range(16))) for meta in (left, right))
        eval_features = core.restrict_paired_features(*(replace(value, design=design)
            for value, design in zip(lfp, evaluation)), plan=plan, check=check)
        eval_counts = core.restrict_paired_activity(*(BoundActivity(design, np.zeros(design.shape))
            for design in evaluation), plan=plan, check=check)
        for feature, counts in zip(eval_features, eval_counts):
            self.assertEqual(feature.design, counts.design)
            np.testing.assert_array_equal(counts.values, np.zeros(counts.design.shape))
        unread = tuple(BoundActivity(meta.design, ForbiddenArray()) for meta in (left, right))
        for wrong in (
            replace(unread[1], design=replace(right.design, axis_ids=right.design.axis_ids[::-1])),
            replace(unread[1], design=replace(right.design, trial_ids=right.design.trial_ids[::-1])),
            replace(unread[1], design=replace(right.design, role="evaluation")),
            replace(unread[1], design=replace(right.design, time_ids=right.design.time_ids[::-1])),
        ):
            with self.assertRaises(ValueError):
                core.restrict_paired_activity(unread[0], wrong, plan=plan, check=check)
        with self.assertRaises(ValueError):
            core.restrict_paired_features(lfp[0], replace(lfp[1], electrode_ids=lfp[1].electrode_ids[::-1]),
                                         plan=plan, check=check)
        with self.assertRaises(ValueError):
            core.restrict_paired_activity(*unread, plan=replace(plan,
                m1=replace(plan.m1, retained_unit_indices=(1, 3))), check=check)
        with self.assertRaises(ValueError):
            core.plan_paired_unit_intersection(replace(left, native_unit_guide=left.native_unit_guide[::-1]),
                                              right, check=check)
        one_unit = metadata("PMd", ((31, 0), (70, 1)), tuple(range(30, 45)))
        with self.assertRaises(core.IntersectionUnavailable):
            core.plan_paired_unit_intersection(left, one_unit, check=check)
        # Invalid values on an unselected native unit cannot be concealed by
        # quietly treating guide restriction as a quality-based count filter.
        invalid = arrays[0].copy()
        invalid[0, 0, 0] = .5
        with self.assertRaises(ValueError):
            core.restrict_paired_activity(BoundActivity(left.design, invalid), activity[1], plan=plan, check=check)


if __name__ == "__main__":
    unittest.main()
