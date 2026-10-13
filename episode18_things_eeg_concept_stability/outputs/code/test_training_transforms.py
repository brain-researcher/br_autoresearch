"""Finite algebra fixtures only; coordinator-owned Slurm execution required."""
import unittest

import numpy as np

from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
from training_transforms import (CommonSensorSpace, DIMENSIONS, EXPANDED_IMAGE,
                                 fit_feature_panel, fit_feature_transform,
                                 fit_residual_whitener, orient_component_signs)


def constant_basis(lags):
    return np.ones((len(lags), 1))


class TrainingTransformTests(unittest.TestCase):
    def test_population_scaling_pca_is_training_only_and_not_whitened(self):
        training = np.array([[0., 0., 7.], [2., 2., 7.], [4., 4., 7.], [6., 6., 7.]])
        saved = training.copy()
        fit = fit_feature_transform(training, training_unit_ids=("a", "b", "c", "d"), requested_rank=32)
        np.testing.assert_array_equal(training, saved)
        np.testing.assert_allclose(fit.mean, [3., 3., 7.])
        np.testing.assert_allclose(fit.population_sd, [np.sqrt(5), np.sqrt(5), 0.])
        self.assertEqual((fit.requested_rank, fit.effective_rank, fit.numerical_rank), (32, 1, 1))
        self.assertEqual(fit.zero_sd_columns, (2,))
        projected = fit.apply(training)
        self.assertAlmostEqual(float(projected.var(ddof=0)), 2.)
        held = np.array([[100., 0., -1000.], [-200., 0., 1000.]])
        expected = ((held[:, :2] - 3.) / np.sqrt(5)) @ fit.loadings[:, :2].T
        np.testing.assert_allclose(fit.apply(held), expected)
        np.testing.assert_array_equal(fit.mean, [3., 3., 7.])
        self.assertFalse(fit.mean.flags.writeable)

    def test_sign_ties_rank_caps_and_zero_variance_are_explicit(self):
        np.testing.assert_array_equal(orient_component_signs([[-1., 1., 0.], [0., -2., 2.]]),
                                      [[1., -1., 0.], [0., 2., -2.]])
        training = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
        fit = fit_feature_transform(training, training_unit_ids=("a", "b", "c", "d"), requested_rank=128)
        self.assertEqual(fit.effective_rank, 3)
        np.testing.assert_allclose(fit.apply(training) @ fit.loadings, training, atol=1e-14)
        zero = fit_feature_transform(np.full((3, 2), .1), training_unit_ids=("a", "b", "c"), requested_rank=64)
        self.assertEqual(zero.effective_rank, 0)
        self.assertEqual(zero.apply([[9., 8.]]).shape, (1, 0))

    def test_duplicate_training_units_unregistered_ranks_and_complex_data_refuse(self):
        for ids, rank, values in ((["a", "a"], 32, [[1.], [2.]]),
                                  (["a", "b"], 12, [[1.], [2.]]),
                                  (["a", "b"], 32, [[1j], [2j]])):
            with self.subTest(ids=ids, rank=rank), self.assertRaises(ValueError):
                fit_feature_transform(values, training_unit_ids=ids, requested_rank=rank)

    def test_expanded_panel_retains_every_family_and_aligned_row_identity(self):
        values = {name: np.zeros((3, DIMENSIONS[name])) for name in EXPANDED_IMAGE}
        values[EXPANDED_IMAGE[0]][:, 0] = [-1., 0., 1.]
        ids = {name: ("a", "b", "c") for name in EXPANDED_IMAGE}
        ranks = {name: 32 for name in EXPANDED_IMAGE}
        fit = fit_feature_panel(values, training_unit_ids=ids, requested_ranks=ranks, panel="expanded_image")
        self.assertEqual(fit.family_order, EXPANDED_IMAGE)
        self.assertEqual([family.effective_rank for family in fit.families], [1, 0, 0, 0, 0, 0])
        self.assertEqual(fit.apply(values, row_ids=ids).shape, (3, 1))
        incomplete = dict(values)
        incomplete.pop(EXPANDED_IMAGE[-1])
        with self.assertRaises(ValueError):
            fit_feature_panel(incomplete, training_unit_ids=ids, requested_ranks=ranks, panel="expanded_image")
        misaligned = dict(ids)
        misaligned[EXPANDED_IMAGE[-1]] = ("b", "a", "c")
        with self.assertRaises(ValueError):
            fit.apply(values, row_ids=misaligned)

    def test_zero_rank_family_integrates_without_removal_or_zero_padding(self):
        transform = fit_feature_transform([[4., 5.], [4., 5.]], training_unit_ids=("a", "b"), requested_rank=32)
        features = transform.apply([[4., 5.], [9., 8.]])
        term = EventTerm(np.array([0., .2]), features, constant_basis, 1, (0., .8))
        part = SequenceDesign("seq", np.array([0., .1, .2, .3]), np.ones((4, 1)), {"required_family": term})
        operator = JointSequenceDesign([part])
        self.assertIn("required_family", part.terms)
        self.assertEqual(part.terms["required_family"].features.shape, (2, 0))
        np.testing.assert_array_equal(operator @ np.array([2.]), np.full(4, 2.))
        np.testing.assert_array_equal(operator.rmatvec(np.ones(4)), [4.])
        np.testing.assert_array_equal(operator.event_contribution([2.], term_name="required_family",
                                      event_masks={"seq": np.ones(2, dtype=bool)}), np.zeros(4))
        with self.assertRaises(ValueError):
            SequenceDesign("empty", np.array([0., .1, .2, .3]), np.empty((4, 0)), {"required_family": term})

    def test_fixed_sensor_basis_preserves_common_average_energy_and_channel_identity(self):
        names = tuple(f"channel_{i}" for i in range(62))
        sensor = CommonSensorSpace(names)
        np.testing.assert_allclose(sensor.basis.T @ sensor.basis, np.eye(61), atol=2e-15)
        np.testing.assert_allclose(sensor.basis @ sensor.basis.T, np.eye(62) - np.ones((62, 62)) / 62., atol=2e-15)
        values = np.arange(124, dtype=float).reshape(2, 62)
        reference = values - values.mean(axis=1, keepdims=True)
        projected = sensor.apply(values, channel_order=names)
        np.testing.assert_allclose(projected @ sensor.basis.T, reference, atol=5e-14)
        np.testing.assert_allclose(np.sum(projected**2, axis=1), np.sum(reference**2, axis=1), rtol=1e-14)
        extra = np.column_stack((values[:, ::-1], np.full(2, 1e8)))
        np.testing.assert_allclose(sensor.apply(extra, channel_order=names[::-1] + ("extra",)), projected)
        np.testing.assert_allclose(sensor.apply(values + [[100.], [-100.]], channel_order=names), projected)
        with self.assertRaises(ValueError):
            sensor.apply(values[:, :61], channel_order=names[:61])

    def test_explicit_training_residual_shrinkage_whitens_covariance_not_response_means(self):
        scale = np.arange(1., 62.) / 20.
        residual_mean = np.arange(61.) / 7.
        residuals = np.vstack((np.diag(scale), -np.diag(scale))) + residual_mean
        options = dict(sample_weights=np.ones(122), center_covariance=True,
                       covariance_denominator=2., shrinkage_target=2. * np.eye(61), shrinkage_strength=.25)
        fit = fit_residual_whitener(residuals, **options)
        empirical = np.diag(scale**2)
        regularized = .75 * empirical + .5 * np.eye(61)
        np.testing.assert_allclose(fit.empirical_covariance, empirical, atol=1e-13)
        np.testing.assert_allclose(fit.regularized_covariance, regularized, atol=1e-13)
        np.testing.assert_allclose(fit.inverse_root @ regularized @ fit.inverse_root, np.eye(61), atol=1e-13)
        np.testing.assert_allclose(fit.residual_mean_for_covariance, residual_mean, atol=1e-13)
        held = np.ones((2, 61)) * 30.
        np.testing.assert_allclose(fit.apply(held), held @ fit.inverse_root, atol=1e-13)
        scaled_weights = dict(options, sample_weights=np.full(122, 2.), covariance_denominator=4.)
        again = fit_residual_whitener(residuals, **scaled_weights)
        np.testing.assert_allclose(again.inverse_root, fit.inverse_root, atol=1e-13)

    def test_singular_or_invalid_whitening_is_not_repaired(self):
        options = dict(sample_weights=[1., 1.], center_covariance=False,
                       covariance_denominator=2., shrinkage_target=np.eye(61), shrinkage_strength=0.)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            fit_residual_whitener(np.zeros((2, 61)), **options)
        for changes in (dict(shrinkage_target=-np.eye(61)), dict(shrinkage_strength=.5 + 1j),
                        dict(covariance_denominator=0.), dict(center_covariance=None)):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                fit_residual_whitener(np.zeros((2, 61)), **dict(options, **changes))


if __name__ == "__main__":
    unittest.main()
