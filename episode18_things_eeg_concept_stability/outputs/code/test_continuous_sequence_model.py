"""Finite algebra checks; run only in the coordinator's compute allocation.

No source events, EEG, images, learned features or simulation study are used.
"""
import unittest

import numpy as np

from continuous_sequence_model import (
    EventTerm, JointSequenceDesign, SequenceDesign, fit_joint_ridge,
)


def basis(lags):
    return np.column_stack((np.ones(len(lags)), lags / .02))


def fixture(sequence_id="s1", empty=False):
    times = np.arange(8, dtype=float) * .004
    events = np.array([]) if empty else np.array([.001, .007, .023])
    features = np.empty((0, 2)) if empty else np.array([[1., 2.], [3., -1.], [-2., .5]])
    nuisance = np.column_stack((np.ones(len(times)), np.arange(len(times)) / 7.))
    return SequenceDesign(sequence_id, times, nuisance, {
        "image": EventTerm(events, features, basis, 2, (0., .02)),
        "added_slot": EventTerm(events, features[:, :1], basis, 2, (0., .02)),
    }), events, features


def dense_reference(part, onsets, features):
    """Direct definition, independent of the operator's cached support slices."""
    result = np.zeros(part.shape)
    result[:, :2] = part.nuisance
    for row, time in enumerate(part.sample_times):
        for onset, feature in zip(onsets, features):
            lag = time - onset
            if 0. <= lag <= .02:
                vector = np.array([1., lag / .02])
                result[row, 2:6] += np.outer(feature, vector).ravel()
                result[row, 6:8] += feature[0] * vector
    return result


class SequenceModelTests(unittest.TestCase):
    def test_actual_subsample_onsets_and_overlap_match_direct_definition(self):
        part, onsets, features = fixture()
        operator = JointSequenceDesign([part])
        expected = dense_reference(part, onsets, features)
        np.testing.assert_allclose(operator @ np.eye(operator.shape[1]), expected,
                                   atol=1e-14, rtol=1e-14)
        # The event at .001 s has not happened at sample zero; it is not rounded.
        np.testing.assert_array_equal(expected[0, 2:], np.zeros(6))
        self.assertNotEqual(expected[1, 3], 0.)

    def test_adjoint_matches_inner_product_and_dense_transpose(self):
        part, onsets, features = fixture()
        operator = JointSequenceDesign([part])
        beta = np.arange(operator.shape[1], dtype=float) - 3.
        samples = np.arange(operator.shape[0], dtype=float) ** 2 / 13.
        expected = dense_reference(part, onsets, features)
        np.testing.assert_allclose(operator.rmatvec(samples), expected.T @ samples,
                                   atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(np.dot(operator @ beta, samples),
                               np.dot(beta, operator.rmatvec(samples)), places=10)

    def test_event_tail_cannot_cross_a_physical_sequence_boundary(self):
        first, _, _ = fixture("first")
        second, _, _ = fixture("second", empty=True)
        operator = JointSequenceDesign([first, second])
        beta = np.zeros(operator.shape[1])
        beta[2:] = 1.
        prediction = operator @ beta
        self.assertTrue(np.any(prediction[:8] != 0.))
        np.testing.assert_array_equal(prediction[8:], np.zeros(8))

    def test_joint_weighted_fit_matches_dense_solve_with_unpenalized_nuisance(self):
        part, onsets, features = fixture()
        operator = JointSequenceDesign([part])
        dense = dense_reference(part, onsets, features)
        targets = np.column_stack((np.arange(8) ** 2 / 9.,
                                   np.array([1., -2., 3., 0., 4., 7., -1., 2.])))
        weights = np.array([1., 2., 0., 1., 3., 1., 1., 2.])
        penalties = np.array([0., 0., .2, .3, .4, .5, .6, .7])
        fit = fit_joint_ridge(operator, targets, sample_weights=weights,
                              penalties=penalties, atol=1e-13, btol=1e-13,
                              conlim=1e12, maxiter=200)
        augmented = np.vstack((np.sqrt(weights)[:, None] * dense,
                               np.diag(np.sqrt(penalties))))
        rhs = np.vstack((np.sqrt(weights)[:, None] * targets, np.zeros((8, 2))))
        expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
        np.testing.assert_allclose(fit.coefficients, expected, atol=1e-9, rtol=1e-9)
        self.assertEqual(len(fit.stop_codes), 2)
        self.assertTrue(all(count <= 200 for count in fit.iterations))

    def test_zero_weight_sample_does_not_affect_fit(self):
        part, _, _ = fixture()
        operator = JointSequenceDesign([part])
        weights = np.ones(8)
        weights[3] = 0.
        target = np.arange(8, dtype=float)
        options = dict(sample_weights=weights, penalties=np.ones(8), atol=1e-12,
                       btol=1e-12, conlim=1e12, maxiter=200)
        original = fit_joint_ridge(operator, target, **options)
        target[3] = 1e9
        changed = fit_joint_ridge(operator, target, **options)
        np.testing.assert_array_equal(original.coefficients, changed.coefficients)

    def test_ablation_removes_only_selected_added_slot_full_support_without_refit(self):
        first, onsets, features = fixture("first")
        second, _, _ = fixture("second")
        operator = JointSequenceDesign([first, second])
        beta = np.column_stack((np.arange(8, dtype=float), np.arange(8) / 3.))
        masks = {"first": np.array([True, False, False]),
                 "second": np.array([False, False, False])}
        removed = operator.event_contribution(beta, term_name="added_slot", event_masks=masks)
        expected = np.zeros((16, 2))
        for row, time in enumerate(first.sample_times):
            lag = time - onsets[0]
            if 0. <= lag <= .02:
                expected[row] = features[0, 0] * np.array([1., lag / .02]) @ beta[6:8]
        np.testing.assert_allclose(removed, expected, atol=1e-13, rtol=1e-13)
        ablated = operator.ablate_events(beta, term_name="added_slot", event_masks=masks)
        np.testing.assert_allclose(ablated, operator @ beta - expected, atol=1e-13, rtol=1e-13)
        np.testing.assert_array_equal(ablated[8:], (operator @ beta)[8:])
        # Its tail overlaps the following event; ablation is not a one-bin deletion.
        self.assertTrue(np.any(removed[3:6] != 0.))

    def test_sequence_schema_and_event_masks_cannot_silently_misalign(self):
        first, _, _ = fixture("first")
        second, _, _ = fixture("first")
        with self.assertRaises(ValueError):
            JointSequenceDesign([first, second])
        different_columns = SequenceDesign("different", first.sample_times,
                                           np.ones((8, 1)), {})
        with self.assertRaises(ValueError):
            JointSequenceDesign([first, different_columns])
        operator = JointSequenceDesign([first])
        with self.assertRaises(ValueError):
            operator.event_contribution(np.ones(8), term_name="added_slot", event_masks={})
        with self.assertRaises(ValueError):
            operator.event_contribution(np.ones(8), term_name="added_slot",
                                        event_masks={"first": np.array([0, 1, 0])})

    def test_fit_rejects_negative_penalty_or_no_retained_samples(self):
        part, _, _ = fixture()
        operator = JointSequenceDesign([part])
        options = dict(atol=1e-12, btol=1e-12, conlim=1e12, maxiter=100)
        with self.assertRaises(ValueError):
            fit_joint_ridge(operator, np.ones(8), sample_weights=np.zeros(8),
                            penalties=np.ones(8), **options)
        with self.assertRaises(ValueError):
            fit_joint_ridge(operator, np.ones(8), sample_weights=np.ones(8),
                            penalties=-np.ones(8), **options)

    def test_iteration_limit_cannot_return_an_unconverged_fit(self):
        part, _, _ = fixture()
        operator = JointSequenceDesign([part])
        with self.assertRaises(RuntimeError):
            fit_joint_ridge(operator, np.arange(8, dtype=float) ** 2,
                            sample_weights=np.ones(8), penalties=np.ones(8),
                            atol=1e-14, btol=1e-14, conlim=1e12, maxiter=1)

    def test_complex_inputs_are_rejected_before_float_conversion(self):
        part, _, _ = fixture()
        operator = JointSequenceDesign([part])
        with self.assertRaisesRegex(ValueError, "real-valued"):
            SequenceDesign("complex", part.sample_times + 1j,
                           part.nuisance, {})
        with self.assertRaisesRegex(ValueError, "real-valued"):
            fit_joint_ridge(operator, np.ones(8) + 1j,
                            sample_weights=np.ones(8), penalties=np.ones(8),
                            atol=1e-12, btol=1e-12, conlim=1e12, maxiter=100)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            operator.matvec(np.ones(8) + 1j)
        with self.assertRaisesRegex(ValueError, "real-valued"):
            operator.event_contribution(np.ones(8) + 1j, term_name="added_slot",
                                        event_masks={"s1": np.zeros(3, dtype=bool)})


if __name__ == "__main__":
    unittest.main()
