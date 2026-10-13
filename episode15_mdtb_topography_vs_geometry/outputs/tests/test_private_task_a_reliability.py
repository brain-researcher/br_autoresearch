"""Known-statistic and private-publication fixtures; no participant payloads."""
import csv
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_private_task_a_reliability as private


IDS = [f"sub-{number:02d}" for number in range(2, 26)]


def known_maps(correlations):
    """Two centered orthogonal equal-norm directions repeated to 208 locations."""
    x = np.tile(np.array([-1., -1., 1., 1.]), 52)
    orthogonal = np.tile(np.array([-1., 1., -1., 1.]), 52)
    first, second = [], []
    for index, r in enumerate(correlations):
        # Distinct offsets/gains expose uncentered cosine or pooled-correlation bugs.
        first.append((index + 1) * x + 100 + index)
        second.append((index + 2) * (r * x + np.sqrt(1 - r*r) * orthogonal) - 40 + index)
    return np.stack((first, second))


class PrivateReliabilityTests(unittest.TestCase):
    def test_known_centered_Pearson_ignores_offsets_and_global_positive_gain(self):
        for expected in (-1., -.6, 0., .6, 1.):
            with self.subTest(expected=expected):
                value = private.reliability_median(known_maps([expected] * 29))
                self.assertAlmostEqual(value, expected, places=14)

    def test_median_of_all29_not_mean_or_positive_only_selection(self):
        values = [-.8] * 15 + [.9] * 14
        median = private.reliability_median(known_maps(values))
        self.assertAlmostEqual(median, -.8, places=14)
        self.assertNotAlmostEqual(median, float(np.mean(values)), places=2)

    def test_unchanged_equal_regressor_then_equal_run_half_maps(self):
        x = np.tile(np.array([-1., -1., 1., 1.]), 52)
        z = np.tile(np.array([-1., 1., -1., 1.]), 52)
        conditions = [f"condition{i:02d}" for i in range(29)]
        plan, mapped = [], {}
        for run in range(1, 17):
            for condition in conditions:
                for duplicate in range(2 if run == 1 else 1):
                    member = f"{run}-{condition}-{duplicate}"
                    plan.append({"run": run, "cond_name": condition, "beta_member": member})
                    base = x + run if run <= 8 else .6*x + .8*z + run
                    mapped[member] = base + ((-4 if duplicate == 0 else 4) if run == 1 else 0)
        halves = private.condition_halves(plan, conditions, lambda row: mapped[row["beta_member"]])
        np.testing.assert_allclose(halves[0], np.tile(x + 4.5, (29, 1)), rtol=0, atol=0)
        np.testing.assert_allclose(halves[1], np.tile(.6*x + .8*z + 12.5, (29, 1)), rtol=0, atol=4e-15)
        self.assertAlmostEqual(private.reliability_median(halves), .6, places=14)

    def test_constant_missing_nonfinite_or_overflow_condition_blocks_all(self):
        good = known_maps([.6] * 29)
        constant = good.copy()
        constant[0, 3] = 7.
        with self.assertRaisesRegex(private.ReliabilityFailure, "centered_spatial_energy"):
            private.reliability_median(constant)
        for bad in (good[:, :-1], good[..., :-1], np.full_like(good, np.nan), np.full_like(good, np.inf)):
            with self.assertRaises(private.ReliabilityFailure):
                private.reliability_median(bad)
        enormous = np.tile(np.array([-1e308, 1e308, -1e308, 1e308]), (2, 29, 52))
        with self.assertRaises((FloatingPointError, private.ReliabilityFailure)):
            private.reliability_median(enormous)

    def test_small_nonzero_variance_has_no_new_floor(self):
        values = known_maps([.6] * 29)
        values -= values.mean(axis=-1, keepdims=True)
        values *= 1e-100
        self.assertAlmostEqual(private.reliability_median(values), .6, places=14)

    def test_partial_cohort_failure_cannot_publish_any_private_rows(self):
        maps = known_maps([.6] * 29)
        calls = []
        def get_halves(pid):
            calls.append(pid)
            if pid == IDS[5]:
                raise private.ReliabilityFailure("synthetic_mapping_failure")
            return maps
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "private"
            target.mkdir(mode=0o700)
            with patch.object(private, "publish_private", wraps=private.publish_private) as publish:
                with self.assertRaises(private.ReliabilityFailure):
                    records = private.collect_private_cohort(IDS, get_halves)
                    private.publish_private(records, IDS, target)
                publish.assert_not_called()
            self.assertEqual(list(target.iterdir()), [])
        self.assertEqual(calls, IDS[:6])

    def test_complete_private_publication_is_restrictive_and_never_overwrites(self):
        records = [{private.PRIVATE_FIELDS[0]: pid, private.PRIVATE_FIELDS[1]: float(index / 24 - .5)}
                   for index, pid in enumerate(IDS)]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "attempt"
            target.mkdir(mode=0o700)
            path = private.publish_private(records, IDS, target)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual([item.name for item in target.iterdir()], [private.PRIVATE_FILENAME])
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, list(private.PRIVATE_FIELDS))
                rows = list(reader)
            self.assertEqual([row["participant_id"] for row in rows], IDS)
            self.assertEqual(len(rows), 24)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                private.publish_private(records, IDS, target)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                private.publish_private(records[:-1], IDS, target)

    def test_failed_publication_removes_temporary_medians(self):
        records = [{private.PRIVATE_FIELDS[0]: pid, private.PRIVATE_FIELDS[1]: .6} for pid in IDS]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(private.os, "link", side_effect=OSError("synthetic publish failure")):
                with self.assertRaises(OSError):
                    private.publish_private(records, IDS, Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_public_report_has_no_reliability_values_or_ranks_and_all24_is_required(self):
        maps = known_maps([.6] * 29)
        report = private.initial_report("/private/attempt/task_a_reliability.csv", 540)
        records = private.collect_private_cohort(IDS, lambda pid: maps)
        self.assertEqual(len(records), 24)
        self.assertEqual(set(records[0]), set(private.PRIVATE_FIELDS))
        public = json.dumps(report, allow_nan=False)
        self.assertNotIn(private.PRIVATE_FIELDS[1], public)
        self.assertNotIn("sub-02", public)
        self.assertNotIn("0.6", public)
        for key in ("minimum_reliability", "maximum_reliability", "median", "ranks", "correlations"):
            self.assertNotIn(key, report)
        self.assertFalse(report["reliability_values_in_public_report"])
        for ids in (IDS[:-1], list(reversed(IDS)), IDS[:-1] + [IDS[0]]):
            with self.assertRaises(ValueError):
                private.collect_private_cohort(ids, lambda pid: maps)


if __name__ == "__main__":
    unittest.main()
