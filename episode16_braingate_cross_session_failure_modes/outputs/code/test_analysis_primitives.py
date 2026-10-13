"""Focused tests for consequential score units and missing-channel semantics."""
import json
from pathlib import Path
import unittest

import numpy as np

from analysis_primitives import (UnresolvedPacket, apply_source_scaler,
    complete_window_means, direction_balanced_score, direction_weights,
    fit_source_scaler)


class AnalysisPrimitiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition = json.loads((Path(__file__).resolve().parents[1] /
            "config/analysis_definition.json").read_text())

    def labels(self):
        angle = (np.arange(8) + 0.5) * np.pi / 4
        return np.column_stack((np.cos(angle), np.sin(angle)))

    def test_score_unit_and_octant_balance(self):
        labels = np.repeat(self.labels(), np.arange(1, 9), axis=0)
        weights = direction_weights(labels, self.definition)
        edges = np.cumsum(np.arange(1, 9))[:-1]
        np.testing.assert_allclose([v.sum() for v in np.split(weights, edges)], 1 / 8)
        self.assertAlmostEqual(direction_balanced_score(labels, np.zeros_like(labels), self.definition), -1)
        self.assertAlmostEqual(direction_balanced_score(labels, -labels, self.definition), -4)
        self.assertEqual(direction_balanced_score(labels, labels, self.definition), 0)

    def test_missing_octant_never_renormalizes_or_replaces_trials(self):
        with self.assertRaises(UnresolvedPacket):
            direction_weights(self.labels()[:-1], self.definition)

    def test_complete_window_and_source_only_axis_preservation(self):
        windows = np.ones((2, 30, 2))
        windows[1, 7, 0] = np.nan
        means = complete_window_means(windows, self.definition)
        self.assertTrue(np.isnan(means[1, 0]))
        self.assertEqual(means[1, 1], 1)
        with self.assertRaises(UnresolvedPacket):
            complete_window_means(windows[:, :-1], self.definition)
        fitted = fit_source_scaler([[1, 5, np.nan], [3, 5, 7]], self.definition)
        np.testing.assert_array_equal(fitted["usable"], [True, False, False])
        actual = apply_source_scaler([[100, 8, 9], [np.nan, 2, 0]], fitted)
        np.testing.assert_allclose(actual, [[98, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(fitted["center"], [2, 0, 0])


if __name__ == "__main__":
    unittest.main()
