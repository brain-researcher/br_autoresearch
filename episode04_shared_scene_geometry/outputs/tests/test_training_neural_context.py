"""Analytic checks of repeat weighting, signed reliability and training isolation."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from preprocess_development_neural import Session
from training_neural_context import (
    chronological_repeats, common_training_images, repeat_moments, roi_context,
)


def session(name, values, image_indices, folds, runs=None, axes=None):
    runs = runs or [1] * len(image_indices)
    axes = axes or list(range(len(image_indices)))
    rows = [{'session': name, 'run': str(run), 'session_beta_axis_index': axis,
             'image_id': str(image)} for run, axis, image in zip(runs, axes, image_indices)]
    return Session(name, np.asarray(values, dtype=np.float32), np.asarray(image_indices),
                   np.asarray(folds), rows)


class TrainingNeuralContextTest(unittest.TestCase):
    def test_unequal_repeats_use_equal_image_variances_and_harmonic_correction(self):
        # Scientific risk: pooling repeat residuals overweights four-repeat images.
        # Var([0,1,2])=1; Var([0,0,0,0])=0, so eta=1/2, h=7/24.
        eta, nc, r = repeat_moments(np.array([0., 1., 2., 0., 0., 0., 0.])[:, None],
                                   np.array([0, 0, 0, 1, 1, 1, 1]),
                                   np.array([0, 1, 0, 0, 1, 0, 1]), 2)
        np.testing.assert_allclose(eta, [.5])
        np.testing.assert_allclose(nc, [2400. / 31.])
        np.testing.assert_allclose(r, [1.])

    def test_nc_boundaries_signed_reliability_and_undefined_voxels_are_distinct(self):
        # Scientific risk: clipping negative consistency or replacing undefined
        # correlation with zero falsely turns noise/constant voxels into evidence.
        values = np.array([[1., 0., 1., np.nan], [1., 2., 1., 0.],
                           [2., 2., 1., 2.], [2., 0., 1., 3.]])
        eta, nc, r = repeat_moments(values, np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), 2)
        np.testing.assert_allclose(eta[:3], [0., 2., 0.])
        np.testing.assert_allclose(nc[:3], [100., 0., 100.])
        np.testing.assert_allclose(r[:2], [1., -1.])
        self.assertTrue(np.isnan(r[2]))
        self.assertTrue(np.isnan(eta[3]) and np.isnan(nc[3]) and np.isnan(r[3]))
        invalid = repeat_moments(np.ones((1, 1)), np.array([0]), np.array([0]), 1)
        self.assertTrue(all(np.isnan(value).all() for value in invalid))

    def test_alternation_uses_session_then_numeric_run_then_original_axis(self):
        # Scientific risk: file/row order or lexicographic runs alter the repeat
        # partition and can turn a three-repeat2:1 split into a selected subset.
        late = session('ses-02', [[2], [3]], [0, 1], [1, 1], runs=[1, 1], axes=[8, 9])
        early = session('ses-01', [[0], [1], [4], [5], [6]], [0, 0, 1, 1, 1], [1] * 5,
                        runs=[10, 2, 2, 1, 2], axes=[0, 1, 6, 5, 4])
        records, counts = chronological_repeats([late, early], [np.ones(2, bool), np.ones(5, bool)],
                                                np.array([True, True]), 0)
        np.testing.assert_array_equal(counts, [3, 4])
        self.assertEqual(records[records[:, 2] == 0, :2].tolist(), [[1, 1], [1, 0], [0, 0]])
        self.assertEqual(records[records[:, 2] == 1, :2].tolist(), [[1, 3], [1, 4], [1, 2], [0, 1]])
        np.testing.assert_array_equal(records[records[:, 2] == 0, 5], [0, 1, 0])
        np.testing.assert_array_equal(records[records[:, 2] == 1, 5], [0, 1, 0, 1])

    def test_diagnostic_evc_intersection_does_not_change_primary_coverage_or_roi_repeats(self):
        # Scientific risk: EVC missingness must not shrink primary images, and a
        # shared context image does not require discarding valid primary repeats.
        primary = np.array([True, True, True])
        evc = np.array([True, False, True])
        common = common_training_images(np.array([1, 1, 0]), 0, primary, evc)
        np.testing.assert_array_equal(common, [True, False, False])
        np.testing.assert_array_equal(primary, [True, True, True])
        example = session('ses-01', [[0], [1], [2], [3], [20], [30]], [0, 0, 0, 0, 1, 2], [1, 1, 1, 1, 1, 0])
        primary_plan, primary_counts = chronological_repeats([example], [np.ones(6, bool)], common, 0)
        evc_plan, evc_counts = chronological_repeats([example], [np.array([1, 1, 0, 1, 1, 1], bool)], common, 0)
        np.testing.assert_array_equal(primary_counts, [4])
        np.testing.assert_array_equal(evc_counts, [3])
        means, scales = np.array([[0.]]), np.array([[1.]])
        primary_eta, _, _ = roi_context([example], means, scales, np.array([0]), primary_plan, 1)
        evc_eta, _, _ = roi_context([example], means, scales, np.array([0]), evc_plan, 1)
        np.testing.assert_allclose(primary_eta, [5. / 3.])
        np.testing.assert_allclose(evc_eta, [7. / 3.])

    def test_evaluation_changes_cannot_change_context_and_saved_scaling_precedes_grouping(self):
        # Scientific risk: a recomputed all-image scale or accidental held-row
        # read leaks evaluation responses into the training reliability context.
        example = session('ses-01', [[10, 20], [12, 24], [14, 28], [16, 32], [99, 99]],
                          [0, 0, 1, 1, 2], [1, 1, 1, 1, 0])
        means = np.array([[10., 20.]])
        scales = np.array([[2., 4.]])
        common = np.array([True, True, False])
        records, _ = chronological_repeats([example], [np.ones(5, bool)], common, 0)
        before = roi_context([example], means, scales, np.array([0, 1]), records, 2, chunk=1)
        # z repeats are[0,1] and[2,3] in both voxels: eta=.5, h=.5, NC=200/3.
        np.testing.assert_allclose(before[0], [.5, .5])
        np.testing.assert_allclose(before[1], [200. / 3., 200. / 3.])
        np.testing.assert_allclose(before[2], [1., 1.])
        changed = copy.deepcopy(example)
        changed.betas[-1] = [np.nan, -np.inf]
        revised_records, _ = chronological_repeats([changed], [np.array([1, 1, 1, 1, 0], bool)], common, 0)
        after = roi_context([changed], means, scales, np.array([0, 1]), revised_records, 2, chunk=2)
        for original, revised in zip(before, after):
            np.testing.assert_array_equal(original, revised)
        np.testing.assert_array_equal(means, [[10., 20.]])
        np.testing.assert_array_equal(scales, [[2., 4.]])


if __name__ == '__main__':
    unittest.main()
