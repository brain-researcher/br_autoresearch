"""One paired trajectory/identity integration; run only in coordinated Slurm."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bandwise_prediction import BANDS, BandwiseLFP
from component_scoring import FrozenDesign
import development_interventions as control


def check():
    pass


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("array accessed before both regional metadata and mapping were bound")


class DevelopmentInterventionTests(unittest.TestCase):
    def test_whole_trajectories_share_one_directional_bijection_and_reject_bad_support_before_values(self):
        # Interleaved and unequal support; trial names, bins and bands are
        # deliberately not sorted. The helper must preserve all supplied order.
        directions = tuple(range(8)) + tuple(range(8)) + (0, 3, 5, 7)
        ids = tuple(f"trial_{19-index:02d}" for index in range(20))
        native = FrozenDesign("Mihili_CO_20140217.mat", "evaluation", tuple(range(8)),
            ids, directions, tuple(range(14, 4, -1)), ("M1:unit0", "M1:unit1"))
        other = replace(native, axis_ids=("PMd:unit0", "PMd:unit1", "PMd:unit2"))
        values = np.arange(20 * 10 * 15 * 9, dtype=float).reshape(20, 10, 15, 9)
        m1 = BandwiseLFP("M1", native, tuple(range(1, 16)), BANDS[::-1], values)
        pmd = BandwiseLFP("PMd", other, tuple(range(81, 96)), BANDS[::-1], 3 * values + 7)
        arguments = dict(session_id=native.session_id, role=native.role, directions=native.directions,
            trial_ids=ids, trial_directions=directions, seed=(6006201, 0, 1, 2), check=check)
        mapping = control.make_within_direction_mapping(**arguments)
        self.assertEqual(mapping, control.make_within_direction_mapping(**arguments))
        self.assertEqual(mapping.trial_ids, ids)
        self.assertEqual(mapping.trial_directions, directions)
        self.assertEqual(sorted(mapping.source_indices), list(range(20)))
        for target, source in enumerate(mapping.source_indices):
            self.assertNotEqual(target, source)
            self.assertEqual(directions[target], directions[source])
            self.assertEqual(mapping.source_trial_ids[target], ids[source])
            # Directions with exactly two trials have a unique mismatch.
            same = [index for index, direction in enumerate(directions) if direction == directions[target]]
            if len(same) == 2:
                self.assertEqual(source, next(index for index in same if index != target))
        changed_m1, changed_pmd = control.apply_paired_lfp_mismatch(m1, pmd, mapping=mapping, check=check)
        self.assertIs(changed_m1.design, native)
        self.assertIs(changed_pmd.design, other)
        self.assertEqual(changed_m1.electrode_ids, tuple(range(1, 16)))
        self.assertEqual(changed_pmd.electrode_ids, tuple(range(81, 96)))
        self.assertEqual(changed_m1.bands, BANDS[::-1])
        self.assertEqual(changed_pmd.bands, BANDS[::-1])
        # Every trajectory coordinate is independently identifiable, so this
        # catches time/electrode/band shuffles and different regional mappings.
        for target, source_id in enumerate(mapping.source_trial_ids):
            source = ids.index(source_id)
            np.testing.assert_array_equal(changed_m1.values[target], values[source])
            np.testing.assert_array_equal(changed_pmd.values[target], 3 * values[source] + 7)
        np.testing.assert_array_equal(m1.values, values)
        for changed in (changed_m1, changed_pmd):
            with self.assertRaises(ValueError):
                changed.values.setflags(write=True)
        unread_m1, unread_pmd = replace(m1, values=ForbiddenArray()), replace(pmd, values=ForbiddenArray())
        for wrong in (
            replace(unread_pmd, design=replace(other, role="training")),
            replace(unread_pmd, design=replace(other, session_id="other_session")),
            replace(unread_pmd, design=replace(other, trial_ids=ids[::-1])),
            replace(unread_pmd, design=replace(other, trial_directions=directions[::-1])),
            replace(unread_pmd, design=replace(other, time_ids=other.time_ids[::-1])),
            replace(unread_pmd, bands=BANDS),
        ):
            with self.assertRaises(ValueError):
                control.apply_paired_lfp_mismatch(unread_m1, wrong, mapping=mapping, check=check)
        crossed = (mapping.source_indices[1], mapping.source_indices[0]) + mapping.source_indices[2:]
        for wrong in (
            replace(mapping, source_indices=tuple(range(20)), source_trial_ids=ids),
            replace(mapping, source_indices=(mapping.source_indices[0],) * 20),
            replace(mapping, source_indices=crossed, source_trial_ids=tuple(ids[index] for index in crossed)),
            replace(mapping, source_trial_ids=ids),
            replace(mapping, role="calibration"),
            replace(mapping, trial_ids=ids[::-1]),
        ):
            with self.assertRaises(ValueError):
                control.apply_paired_lfp_mismatch(unread_m1, unread_pmd, mapping=wrong, check=check)
        # One trial in direction0 remains an explicit unsupported control;
        # retaining other directions cannot silently turn it into a subset.
        retain = tuple(index for index in range(20) if index not in (8, 16))
        with self.assertRaises(control.TrialMismatchUnavailable):
            control.make_within_direction_mapping(**{**arguments,
                "trial_ids": tuple(ids[index] for index in retain),
                "trial_directions": tuple(directions[index] for index in retain)})
        def stopped():
            raise TimeoutError("caller cancellation")
        with self.assertRaises(TimeoutError):
            control.apply_paired_lfp_mismatch(unread_m1, unread_pmd, mapping=mapping, check=stopped)


if __name__ == "__main__":
    unittest.main()
