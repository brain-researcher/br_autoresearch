"""Focused utility, chosen-read, fixed-rank and common-score checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from allocation_reader import AllocationTrajectory, FittingSource
from development_trial_allocation import score_session
from ridge_training import new_progress
from trial_allocation import POLICIES, choose_direction, frozen_numeric_rank, fit_frozen_rank


class TrialAllocationTest(unittest.TestCase):
    def test_reducible_value_and_total_error_choose_different_legal_directions(self):
        counts = np.array([8, 4, 4, 4, 4, 4, 4, 4])
        labels = np.repeat(np.arange(8), counts)
        rows = []
        for d, n in enumerate(counts):
            values = np.arange(n, dtype=float)-(n-1)/2
            rows.extend(values*np.sqrt((16 if d == 0 else 4)/np.var(values, ddof=1)))
        y = np.array(rows)[:, None]
        value, detail = choose_direction(y, labels, np.ones(1), counts, "mean_value")
        error, _ = choose_direction(y, labels, np.ones(1), counts, "mean_error")
        self.assertEqual(value, 1)
        self.assertEqual(error, 0)
        self.assertAlmostEqual(detail["pooled_variance"], 7.)
        np.testing.assert_allclose(detail["utility"], [13.3/72]+[5.5/20]*7, atol=1e-14)
        for policy in POLICIES:
            chosen, _ = choose_direction(np.zeros_like(y), labels, np.ones(1), counts, policy)
            self.assertEqual(chosen, 1)

    def test_trajectory_reads_only_chosen_next_row_and_fixed_channels(self):
        pools = [list(range(d*12, (d+1)*12)) for d in range(8)]
        calls = []
        def read(ids, channels):
            calls.append((ids, channels))
            return np.full((len(ids), 6), ids[0]), np.full((len(ids), 2), ids[0])
        source = SimpleNamespace(role={"fitting_pool_by_direction": pools}, read=read)
        t = AllocationTrajectory(source, [9, 1], np.zeros((32, 6)), np.zeros((32, 2)))
        self.assertEqual(calls, [])
        self.assertEqual(t.acquire(3), 40)
        self.assertEqual(calls, [([40], [9, 1])])
        x, y, labels, ids = t.observed()
        self.assertEqual(ids, [v for d, q in enumerate(pools) for v in q[:5 if d == 3 else 4]])
        for _ in range(7):
            t.acquire(3)
        before = len(calls)
        with self.assertRaises(ValueError):
            t.acquire(3)
        self.assertEqual(len(calls), before)
        # The trusted low-level entrypoint rejects an evaluation ID without
        # touching a source field/handle.
        reader = object.__new__(FittingSource)
        reader.allowed, reader.source = set(range(96)), {"common_physical_channels": list(range(1, 97))}
        with self.assertRaises(ValueError):
            reader.read([96], [9, 1])

    def test_full_rank_is_an_initial_numeric_cap_with_no_later_cv(self):
        rank = frozen_numeric_rank(SimpleNamespace(rank="full"), 40, 40)
        self.assertEqual(rank, 24)
        rng = np.random.default_rng(80403)
        x, y, labels = rng.normal(size=(64, 40)), rng.normal(size=(64, 40)), np.repeat(np.arange(8), 8)
        progress = new_progress()
        model = fit_frozen_rank(x, y, labels, .01, rank, progress)
        self.assertEqual(model.rank, 24)
        self.assertEqual(model.alpha, .01)
        self.assertLessEqual(np.linalg.matrix_rank(model.coefficients), 24)
        self.assertEqual(progress["inner_ridge_fits"], 0)
        self.assertEqual(progress["outer_training_refits"], 1)

    def test_scoring_uses_common_initial_scale_despite_changed_fitting_statistics(self):
        labels = np.repeat(np.arange(8), 5)
        y = np.arange(80, dtype=float).reshape(40, 2)/10
        prediction = y*.8
        fixed_mean, fixed_sd = np.array([.4, -.2]), np.array([2., 3.])
        models = []
        for policy_index, policy in enumerate(POLICIES):
            for total in (32, 48, 64, 80, 96):
                for count in (8, 16, 32, 96):
                    for seed in range(1 if count == 96 else 5):
                        fitted = SimpleNamespace(rank=1, alpha=.01,
                            statistics=SimpleNamespace(target_mean=np.full(2, policy_index+99.), target_sd=np.full(2, total)))
                        models.append((policy, total, count, seed, tuple(range(1, count+1)), fitted,
                                       fixed_mean, fixed_sd, (total//8,)*8))
        with patch("development_trial_allocation.predict_capacity", return_value=prediction):
            panels = score_session(models, np.zeros((40, 288)), y, labels, list(range(1, 97)))
        expected = 1-np.square((y-prediction)/fixed_sd).sum()/np.square((y-fixed_mean)/fixed_sd).sum()
        for policy in POLICIES:
            self.assertEqual(len(panels[policy]["cells"]), 20)
            for cell in panels[policy]["cells"]:
                self.assertAlmostEqual(cell["Q_common"], expected)
                self.assertEqual(cell["delta_Q_common_from_balanced"], 0.)


if __name__ == "__main__":
    unittest.main()
