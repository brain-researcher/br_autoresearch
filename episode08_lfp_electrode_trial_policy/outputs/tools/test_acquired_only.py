"""Focused acquisition-view and conditioning-history checks; no source payloads."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import acquired_only_reader as ar
from acquired_rank_training import conditioned_folds, validate_conditioned_folds


class AcquiredOnlyTest(unittest.TestCase):
    def test_post_initial_view_is_selected_channels_and_exact_next16trials(self):
        pools = [list(range(d*12, (d+1)*12)) for d in range(8)]
        role = {"fitting_pool_by_direction": pools,
                "nested_training_sets": {"96": {"training_trials": sum(pools, [])}},
                "evaluation_trials": list(range(96, 136)), "filename": "Mihili_CO_20140217.mat", "session_role": "development"}
        source = {"common_physical_channels": list(range(1, 97)), "filename": role["filename"], "session_role": "development"}
        reads = []
        def fake_read(path, source, role, records, ids, channels, check, exposure, reference=None):
            reads.append((list(ids), list(channels)))
            return np.ones((len(ids), 3*len(channels))), np.ones((len(ids), 2)), ("guide", "units")
        trajectory = ar.Trajectory(Path(role["filename"]), source, role, {}, [9, 1], lambda: None, None)
        trajectory.initialize_selected(np.zeros((32, 288)), np.zeros((32, 2)), list(range(1, 97)), ("guide", "units"))
        with patch.object(ar, "read_trials", side_effect=fake_read):
            trajectory.acquire_to(48)
            with self.assertRaises(ValueError):
                trajectory.acquire_to(80)
        expected = [t for pool in pools for t in pool[4:6]]
        self.assertEqual(reads, [(expected, [9, 1])])
        self.assertEqual(trajectory.observed()[0].shape, (48, 6))
        with patch.object(ar.h5py, "File") as forbidden:
            with self.assertRaises(ValueError):
                ar.read_trials(Path(role["filename"]), source, role, {}, [96], [1], lambda: None, None)
            forbidden.assert_not_called()
        random = ar.Trajectory(Path(role["filename"]), source, role, {}, [9, 1], lambda: None, None)
        with patch.object(ar, "read_trials", side_effect=fake_read):
            random.initialize_random()
        self.assertEqual(reads[-1], (ar.balanced_ids(role, 32), [9, 1]))

    def test_all_initializer_rows_remain_in_every_later_fit(self):
        for per_direction in (6, 8, 10, 12):
            labels = np.repeat(np.arange(8), per_direction)
            positions = np.tile(np.arange(per_direction)-4, 8)
            mask = positions < 0
            initial, folds = conditioned_folds(labels, mask, positions)
            self.assertEqual(len(initial), 32)
            validation = []
            for fit, valid in folds:
                self.assertTrue(set(initial) <= set(fit))
                self.assertFalse(set(initial) & set(valid))
                self.assertGreater(len(valid), 0)
                validation.extend(valid)
            self.assertEqual(sorted(validation), list(np.flatnonzero(~mask)))
            broken = [(f[f != initial[0]], v) for f, v in folds]
            with self.assertRaises(ValueError):
                validate_conditioned_folds(len(labels), initial, broken)


if __name__ == "__main__":
    unittest.main()
