"""Independent expected values and leakage/missing-support counterexamples."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from preprocess_development_neural import (
    Session, SESSIONS, completed_session_records, image_eligibility, mean_pattern_block,
    presentation_validity, repeat_counts, retained_roi_support, training_statistics,
)


def session(name, values, image_indices, folds):
    return Session(name, np.asarray(values, dtype=np.float32), np.asarray(image_indices), np.asarray(folds), [])


class NeuralPreprocessingTest(unittest.TestCase):
    def test_evaluation_changes_cannot_change_training_statistics_or_voxel_support(self):
        values = np.arange(12)[None, :] + np.array([1., 2., 3., 20., 30.])[:, None]
        sessions = [session('a', values, [0, 1, 2, 3, 4], [1, 1, 2, 0, 0]),
                    session('b', values + 10, [0, 1, 2, 3, 4], [1, 1, 2, 0, 0])]
        mean, scale, keep, *_ = training_statistics(sessions, 0, chunk_voxels=2)
        np.testing.assert_allclose(mean[0], np.arange(12) + 2)
        np.testing.assert_allclose(mean[1], np.arange(12) + 12)
        np.testing.assert_allclose(scale, np.sqrt(2 / 3))
        np.testing.assert_array_equal(keep, np.ones(12, dtype=bool))
        changed = copy.deepcopy(sessions)
        changed[0].betas[3:] = np.nan
        changed[1].betas[3:] = -np.inf
        revised = training_statistics(changed, 0, chunk_voxels=5)
        for original, revised_array in zip((mean, scale, keep), revised[:3]):
            np.testing.assert_array_equal(original, revised_array)

    def test_zero_scale_and_nonfinite_training_are_intersected_across_sessions(self):
        first = session('a', [[1, 0, 1, 5], [2, 1, np.nan, 6], [3, 2, 3, 7], [8, 9, 10, np.nan]],
                        [0, 1, 2, 3], [1, 1, 2, 0])
        second = session('b', [[3, 0, 1, 5], [4, 0, 2, 6], [5, 0, 3, 7], [8, 0, 10, 11]],
                         [0, 1, 2, 3], [1, 1, 2, 0])
        mean, scale, keep, finite, positive, _ = training_statistics([first, second], 0, chunk_voxels=2)
        np.testing.assert_array_equal(keep, [True, False, False, True])
        np.testing.assert_array_equal(finite, [True, True, False, True])
        self.assertEqual(scale[1, 1], 0)
        self.assertTrue(np.isnan(mean[0, 2]))
        self.assertTrue(positive[3])  # A held-row NaN cannot remove its voxel.

    def test_primary_validity_is_common_across_sectors_and_independent_of_evc(self):
        values = np.arange(12)[None, :] + np.array([1., 2., 3., 20., 30.])[:, None]
        example = session('a', values, [0, 1, 2, 3, 4], [1, 1, 2, 0, 0])
        membership = np.zeros((12, 4), dtype=bool)
        for roi in range(4):
            membership[3 * roi:3 * (roi + 1), roi] = True
        mean, scale, keep, *_ = training_statistics([example], 0)
        example.betas[3, 6] = np.nan  # Dorsal failure invalidates the primary presentation.
        example.betas[4, 9] = np.nan  # EVC failure cannot invalidate primary.
        primary, evc = presentation_validity([example], mean, scale, retained_roi_support(membership, keep), 2)
        np.testing.assert_array_equal(primary[0], [True, True, True, False, True])
        np.testing.assert_array_equal(evc[0], [True, True, True, True, False])
        altered = example.betas.copy()
        example.betas[:, 9:] = np.nan
        still_primary, _ = presentation_validity([example], mean, scale, retained_roi_support(membership, keep))
        np.testing.assert_array_equal(still_primary[0], primary[0])
        example.betas = altered

    def test_unequal_repeats_use_every_valid_presentation_and_evc_cannot_rescue_images(self):
        sessions = [session('a', [[1, 2], [3, 6], [2, 4], [4, 8], [9, 18]], [0, 0, 1, 1, 2], [1] * 5),
                    session('b', [[5, 10], [6, 12], [8, 16]], [0, 1, 1], [1] * 3)]
        valid = [np.ones(5, dtype=bool), np.ones(3, dtype=bool)]
        counts = repeat_counts(sessions, valid, 3)
        np.testing.assert_array_equal(counts, [3, 4, 1])
        primary, complete, evc = image_eligibility(counts, np.array([1, 4, 4]))
        np.testing.assert_array_equal(primary, [True, True, False])
        np.testing.assert_array_equal(complete, [False, True, False])
        np.testing.assert_array_equal(evc, [False, True, False])
        mean = np.zeros((2, 2))
        scale = np.array([[1., 2.], [1., 2.]])
        patterns = mean_pattern_block(sessions, mean, scale, np.array([0, 1]), valid, counts, primary)
        np.testing.assert_array_equal(patterns, [[3, 3], [5, 5], [np.nan, np.nan]])
        # Same physical voxel can have different means for different ROI repeat support.
        evc_valid = [valid[0], np.array([False, True, True])]
        evc_counts = repeat_counts(sessions, evc_valid, 3)
        _, _, evc_eligible = image_eligibility(counts, evc_counts)
        evc_patterns = mean_pattern_block(sessions, mean, scale, np.array([0]), evc_valid, evc_counts, evc_eligible)
        self.assertEqual(evc_patterns[0, 0], 2)
        self.assertEqual(patterns[0, 0], 3)

    def test_insufficient_training_session_is_unestimable_not_a_nonfinite_observation(self):
        example = session('a', [[1, 2], [8, 9]], [0, 1], [1, 0])
        mean, scale, keep, finite, _, counts = training_statistics([example], 0)
        np.testing.assert_array_equal(counts, [1])
        self.assertFalse(keep.any())
        self.assertTrue(np.isnan(mean).all() and np.isnan(scale).all())
        self.assertFalse(finite.any())  # Unassessed masks; no finite-data certificate.

    def test_partial_or_wrong_participant_handoff_is_rejected_before_beta_access(self):
        report = {'participant': 'sub-01', 'complete_participant_handoff': True, 'errors': [],
                  'sessions': [{'session': name, 'complete': True} for name in SESSIONS]}
        self.assertEqual(len(completed_session_records(report, 'sub-01')), 30)
        with self.assertRaises(ValueError):
            completed_session_records(report, 'sub-03')
        changed = copy.deepcopy(report)
        changed['sessions'][13]['complete'] = False
        with self.assertRaises(ValueError):
            completed_session_records(changed, 'sub-01')
        changed['sessions'][13]['complete'] = True
        changed['sessions'].pop()
        with self.assertRaises(ValueError):
            completed_session_records(changed, 'sub-01')


if __name__ == '__main__':
    unittest.main()
