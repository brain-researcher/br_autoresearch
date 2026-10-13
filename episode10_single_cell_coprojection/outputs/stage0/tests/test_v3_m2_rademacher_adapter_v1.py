#!/usr/bin/env python3
"""Target-free tests for the frozen EP10 v3 Rademacher adapter."""

from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_rademacher_adapter_v1 as rad  # noqa: E402


DIRECT_SEED = 2026092801


def _global_rng_state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _decoded_sign(words: np.ndarray, row: int, column: int) -> int:
    flat_index = row * rad.COLUMN_COUNT + column
    word = int(words[flat_index // 64])
    bit = (word >> (flat_index % 64)) & 1
    return 1 if bit else -1


class RademacherAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.cells,
            cls.moments,
            _,
            cls.candidates,
            cls.geometry,
            _,
        ) = m1.generate_synthetic_dataset(
            groups=1,
            rows_per_group=17,
            seed=2026092801,
        )
        cls.physical_pairs = m2.PairFeatureLibrary(cls.candidates)
        cls.physical_transform, cls.physical_prepared = m2.fit_residual_transform(
            cls.cells,
            cls.moments,
            cls.candidates,
            cls.geometry,
            cls.physical_pairs,
            expected_rank=rad.EXPECTED_PROJECTED_RANK,
        )
        cls.physical_snapshots = {
            "soma_mean": cls.physical_transform.design_transform.soma_mean.copy(),
            "soma_scale": cls.physical_transform.design_transform.soma_scale.copy(),
            "geometry_scales": cls.physical_transform.geometry_scales.copy(),
            "projection": cls.physical_transform.projection_f_to_h.copy(),
            "pair_RMS": cls.physical_transform.pair_residual_scales.copy(),
            "range_basis": cls.physical_transform.range_basis.copy(),
            "range_projector": cls.physical_transform.range_projector.copy(),
        }
        cls.physical_feature_before = m2.residual_pair_batch(
            cls.physical_prepared,
            8,
            0,
            64,
            cls.candidates,
            cls.geometry,
            cls.physical_pairs,
        ).copy()
        cls.seed_route = rad.RademacherSeedRoute(
            "target_free_qualification",
            DIRECT_SEED,
        )
        cls.features = rad.RademacherFeatureLibrary(
            cls.candidates,
            cls.seed_route,
        )
        cls.fitted = rad.fit_rademacher_fold(
            cls.cells,
            cls.moments,
            cls.candidates,
            cls.geometry,
            cls.features,
            cls.physical_prepared,
            cls.physical_pairs,
            expected_rank=rad.EXPECTED_PROJECTED_RANK,
        )
        cls.fixed_m1_theta = 0.15 * m1.planted_parameters()

        cls.validation_cells = tuple(
            dataclasses.replace(
                cell,
                identifier=f"rademacher_validation_{index:02d}",
                group="synthetic_validation_group",
                soma_um=tuple(value + 25.0 * (index + 1) for value in cell.soma_um),
            )
            for index, cell in enumerate(cls.cells[:5])
        )
        cls.validation_moments = tuple(
            cls.geometry.uniform_moments(cell) for cell in cls.validation_cells
        )
        cls.validation_prepared = rad.apply_rademacher_validation(
            cls.fitted,
            cls.validation_cells,
            cls.validation_moments,
        )

    def test_exact_dictionary_shape_order_centering_and_immutability(self) -> None:
        matrix = self.features.matrix
        self.assertEqual(matrix.shape, (rad.ROW_COUNT, rad.COLUMN_COUNT))
        self.assertTrue(matrix.flags.c_contiguous)
        self.assertFalse(matrix.flags.writeable)
        self.assertEqual(
            self.features.all_counts(),
            {k: math.comb(m1.PANEL_SIZE, k) for k in range(m1.PANEL_SIZE + 1)},
        )
        self.assertEqual(sum(self.features.all_counts().values()), rad.ROW_COUNT)

        offset = 0
        for k in range(m1.PANEL_SIZE + 1):
            block = self.features.basis(k).indicators
            count = math.comb(m1.PANEL_SIZE, k)
            self.assertEqual(block.shape, (count, rad.COLUMN_COUNT))
            self.assertTrue(np.shares_memory(block, matrix))
            self.assertFalse(block.flags.writeable)
            np.testing.assert_array_equal(block, matrix[offset : offset + count])
            block_mean = np.mean(block, axis=0)
            # The prospectively frozen operation is one binary64 subtraction
            # of the exact integer-sum/count mean, not a second recentering.
            self.assertLessEqual(float(np.max(np.abs(block_mean))), 5.0e-15)
            offset += count
        self.assertEqual(offset, rad.ROW_COUNT)
        np.testing.assert_array_equal(self.features.basis(0).indicators, np.zeros((1, 120)))
        np.testing.assert_array_equal(
            self.features.basis(16).indicators, np.zeros((1, 120))
        )
        with self.assertRaises(ValueError):
            matrix[0, 0] = 1.0
        with self.assertRaises(ValueError):
            matrix.setflags(write=True)

        words = np.random.PCG64(DIRECT_SEED).random_raw(rad.RAW_WORD_COUNT)
        start = 0
        for k in range(17):
            count = math.comb(16, k)
            sampled_rows = tuple(sorted({0, count // 2, count - 1}))
            for column in (0, 63, 119):
                signs = [
                    _decoded_sign(words, start + row, column)
                    for row in range(count)
                ]
                integer_sum = sum(signs)
                for local_row in sampled_rows:
                    expected = float(signs[local_row]) - (
                        float(integer_sum) / float(count)
                    )
                    self.assertEqual(
                        matrix[start + local_row, column],
                        expected,
                    )
            start += count
        self.assertEqual(start, rad.ROW_COUNT)

        statistics = self.features.statistics(8)
        for array in (
            statistics.mean,
            statistics.covariance,
            statistics.contrast_cross_covariance,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_seed_routes_repeatability_separation_and_global_rng_isolation(self) -> None:
        direct = rad.RademacherSeedRoute("observed", DIRECT_SEED)
        expected_direct = np.random.PCG64(DIRECT_SEED).random_raw(16)
        np.testing.assert_array_equal(
            direct.bit_generator().random_raw(16), expected_direct
        )

        null = rad.RademacherSeedRoute(
            "null_replay",
            DIRECT_SEED,
            null_replicate_seed=2026093001,
            valid_index=17,
        )
        expected_sequence = np.random.SeedSequence(
            2026093001,
            spawn_key=(3, 0, 17, 3),
        )
        np.testing.assert_array_equal(
            null.bit_generator().random_raw(16),
            np.random.PCG64(expected_sequence).random_raw(16),
        )
        same_null_draws_different_authenticated_trial_seed = rad.RademacherSeedRoute(
            "null_replay",
            DIRECT_SEED + 99,
            null_replicate_seed=2026093001,
            valid_index=17,
        )
        np.testing.assert_array_equal(
            null.bit_generator().random_raw(16),
            same_null_draws_different_authenticated_trial_seed.bit_generator().random_raw(16),
        )
        separated = rad.RademacherSeedRoute(
            "null_replay",
            DIRECT_SEED,
            null_replicate_seed=2026093001,
            valid_index=18,
        )
        self.assertFalse(
            np.array_equal(
                null.bit_generator().random_raw(16),
                separated.bit_generator().random_raw(16),
            )
        )

        np.random.seed(913)
        before = np.random.get_state()
        _ = null.bit_generator().random_raw(64)
        after = np.random.get_state()
        self.assertTrue(_global_rng_state_equal(before, after))

        before_dictionary = np.random.get_state()
        repeated = rad.RademacherFeatureLibrary(self.candidates, self.seed_route)
        after_dictionary = np.random.get_state()
        self.assertTrue(_global_rng_state_equal(before_dictionary, after_dictionary))
        np.testing.assert_array_equal(repeated.matrix, self.features.matrix)
        different = rad.RademacherFeatureLibrary(
            self.candidates,
            rad.RademacherSeedRoute(
                "target_free_qualification",
                DIRECT_SEED + 1,
            ),
        )
        self.assertFalse(
            np.array_equal(
                different.matrix[1:17],
                self.features.matrix[1:17],
            )
        )

        with self.assertRaises(rad.RademacherAdapterError):
            rad.RademacherSeedRoute("null_replay", DIRECT_SEED)
        with self.assertRaises(rad.RademacherAdapterError):
            rad.RademacherSeedRoute(
                "observed",
                DIRECT_SEED,
                null_replicate_seed=2026093001,
                valid_index=17,
            )

    def test_noise_projection_reuses_physical_objects_and_has_stable_rank_104(self) -> None:
        fitted = self.fitted
        transform = fitted.transform
        physical = self.physical_transform
        self.assertIs(fitted.features, self.features)
        self.assertIs(transform.design_transform, physical.design_transform)
        self.assertIs(transform.geometry_scales, physical.geometry_scales)
        self.assertIs(transform.pair_residual_scales, physical.pair_residual_scales)
        self.assertIs(transform.range_basis, physical.range_basis)
        self.assertIs(transform.range_projector, physical.range_projector)
        self.assertIs(fitted.physical_prepared, self.physical_prepared)
        self.assertIs(fitted.prepared.m1_data, self.physical_prepared.m1_data)
        self.assertIs(fitted.prepared.m1_data.design_transform, physical.design_transform)
        np.testing.assert_array_equal(
            fitted.prepared.m1_data.geometry_scales,
            physical.geometry_scales,
        )
        np.testing.assert_array_equal(
            fitted.prepared.m1_data.geometry_centers,
            self.physical_prepared.m1_data.geometry_centers,
        )

        diagnostics = transform.diagnostics
        self.assertLessEqual(
            diagnostics["G_FR_projection_maximum_absolute_orthogonality_error"],
            m2.PROJECTION_TOLERANCE,
        )
        self.assertTrue(diagnostics["rank_stable"])
        self.assertEqual(diagnostics["rank"], rad.EXPECTED_PROJECTED_RANK)
        self.assertEqual(
            diagnostics["rank_at_tau_over_10_tau_ten_tau"],
            {"tau_over_10": 104, "tau": 104, "ten_tau": 104},
        )
        self.assertTrue(diagnostics["physical_pair_residual_RMS_reused"])
        self.assertTrue(diagnostics["physical_range_basis_reused"])
        self.assertTrue(diagnostics["physical_range_projector_reused"])
        self.assertTrue(diagnostics["physical_geometry_centers_reused"])
        self.assertTrue(diagnostics["G_FF_same_pass_physical_provenance"])
        self.assertTrue(
            diagnostics["G_FF_pseudoinverse_same_pass_physical_provenance"]
        )
        self.assertFalse(diagnostics["seed_route"]["spawn_key"])

        null_gram = (
            np.eye(rad.COLUMN_COUNT) - transform.range_projector
        ) @ transform.scaled_residual_gram
        self.assertLess(float(np.max(np.abs(null_gram))), 2.0e-10)

    def test_validation_is_training_frozen_and_beta_zero_exactly_nests_m1(self) -> None:
        transform = self.fitted.transform
        projection_before = transform.projection_f_to_h.copy()
        rms_before = transform.pair_residual_scales.copy()
        projector_before = transform.range_projector.copy()
        first = rad.apply_rademacher_validation(
            self.fitted,
            self.validation_cells,
            self.validation_moments,
        )
        second = rad.apply_rademacher_validation(
            self.fitted,
            self.validation_cells,
            self.validation_moments,
        )
        self.assertIs(first.transform, transform)
        self.assertIs(second.transform, transform)
        self.assertIs(first.m1_data.design_transform, transform.design_transform)
        np.testing.assert_array_equal(first.m1_data.x, second.m1_data.x)
        np.testing.assert_array_equal(transform.projection_f_to_h, projection_before)
        np.testing.assert_array_equal(transform.pair_residual_scales, rms_before)
        np.testing.assert_array_equal(transform.range_projector, projector_before)

        noise_features = m2.residual_pair_batch(
            first,
            4,
            0,
            min(64, self.candidates.basis(4).count),
            self.candidates,
            self.geometry,
            self.features,
        )
        self.assertEqual(noise_features.shape[1], rad.COLUMN_COUNT)
        self.assertTrue(np.all(np.isfinite(noise_features)))

        zero = np.zeros(rad.COLUMN_COUNT, dtype=float)
        noise_zero = m2.evaluate_m2(
            zero,
            self.fixed_m1_theta,
            first,
            self.candidates,
            self.geometry,
            self.features,
            need_predictions=True,
        )
        m1_zero = m1.evaluate_model(
            self.fixed_m1_theta,
            first.m1_data,
            self.candidates,
            self.geometry,
            0.0,
            need_hessian=False,
            need_predictions=True,
        )
        assert noise_zero.log_probability_nats is not None
        assert noise_zero.marginals is not None
        assert m1_zero.log_probability_nats is not None
        assert m1_zero.marginals is not None
        np.testing.assert_array_equal(
            noise_zero.log_probability_nats,
            m1_zero.log_probability_nats,
        )
        np.testing.assert_array_equal(noise_zero.marginals, m1_zero.marginals)

    def test_adapter_leaves_physical_pair_path_bit_identical(self) -> None:
        physical = self.physical_transform
        for label, current in (
            ("soma_mean", physical.design_transform.soma_mean),
            ("soma_scale", physical.design_transform.soma_scale),
            ("geometry_scales", physical.geometry_scales),
            ("projection", physical.projection_f_to_h),
            ("pair_RMS", physical.pair_residual_scales),
            ("range_basis", physical.range_basis),
            ("range_projector", physical.range_projector),
        ):
            np.testing.assert_array_equal(current, self.physical_snapshots[label])
        physical_feature_after = m2.residual_pair_batch(
            self.physical_prepared,
            8,
            0,
            64,
            self.candidates,
            self.geometry,
            self.physical_pairs,
        )
        np.testing.assert_array_equal(
            physical_feature_after,
            self.physical_feature_before,
        )

    def test_same_pass_physical_gram_rejects_fresh_agreeing_moment_recomputations(
        self,
    ) -> None:
        changed_covariance = self.moments[0].raw_joint_covariance.copy()
        changed_covariance[0, 0] += 0.125
        altered = (
            dataclasses.replace(
                self.moments[0],
                raw_joint_covariance=changed_covariance,
            ),
            *self.moments[1:],
        )
        first, _ = m1.fold_gram(
            self.cells, altered, tuple(range(len(self.cells)))
        )
        second, _ = m1.fold_gram(
            self.cells, altered, tuple(range(len(self.cells)))
        )
        np.testing.assert_array_equal(first, second)
        physical = m2._reauthenticate_physical_fold_provenance(
            self.physical_prepared
        )
        self.assertFalse(np.array_equal(first, physical.gram_ff))
        with self.assertRaisesRegex(
            rad.RademacherAdapterError,
            "same-pass provenance",
        ):
            rad.fit_rademacher_fold(
                self.cells,
                altered,
                self.candidates,
                self.geometry,
                self.features,
                self.physical_prepared,
                self.physical_pairs,
            )

    def test_unregistered_physical_preparation_fails_closed(self) -> None:
        forged = dataclasses.replace(self.physical_prepared)
        with self.assertRaisesRegex(
            rad.RademacherAdapterError,
            "same-pass provenance",
        ):
            rad.fit_rademacher_fold(
                self.cells,
                self.moments,
                self.candidates,
                self.geometry,
                self.features,
                forged,
                self.physical_pairs,
            )

    def test_registered_noise_projection_uses_exact_stored_projector(self) -> None:
        rng = np.random.default_rng(2026092901)
        rows = rng.normal(size=(7, rad.COLUMN_COUNT))
        vector = rng.normal(size=rad.COLUMN_COUNT)
        transform = self.fitted.transform
        projector = transform.range_projector
        basis = transform.range_basis

        expected_rows = rows @ projector
        legacy_rows = (rows @ basis) @ basis.T
        self.assertFalse(np.array_equal(expected_rows, legacy_rows))
        np.testing.assert_array_equal(
            m2._range_project_rows(rows, transform), expected_rows
        )

        expected_vector = projector @ vector
        legacy_vector = basis @ (basis.T @ vector)
        self.assertFalse(np.array_equal(expected_vector, legacy_vector))
        np.testing.assert_array_equal(
            m2._range_project_vector(vector, transform), expected_vector
        )

        forged = dataclasses.replace(
            transform,
            range_projector=np.zeros_like(projector),
        )
        with self.assertRaisesRegex(m2.M2Error, "registered provenance"):
            m2._range_project_rows(rows, forged)
        with self.assertRaisesRegex(m2.M2Error, "registered provenance"):
            m2._range_project_vector(vector, forged)

    def test_dictionary_route_and_matrix_relabelling_fail_closed(self) -> None:
        different_route = rad.RademacherSeedRoute(
            "target_free_qualification", DIRECT_SEED + 99
        )
        with self.assertRaisesRegex(rad.RademacherAdapterError, "immutable"):
            self.features._seed_route = different_route

        original_route = self.features.seed_route
        object.__setattr__(self.features, "_seed_route", different_route)
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2._range_project_vector(
                    np.ones(rad.COLUMN_COUNT), self.fitted.transform
                )
        finally:
            object.__setattr__(self.features, "_seed_route", original_route)

        original_matrix = self.features.matrix
        replacement = original_matrix.copy()
        replacement.setflags(write=False)
        object.__setattr__(self.features, "_matrix", replacement)
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2.residual_pair_batch(
                    self.fitted.prepared,
                    8,
                    0,
                    8,
                    self.candidates,
                    self.geometry,
                    self.features,
                )
        finally:
            object.__setattr__(self.features, "_matrix", original_matrix)

    def test_rademacher_statistics_mutation_and_replacement_fail_before_fit(
        self,
    ) -> None:
        statistics = self.features.statistics(8)
        original_cross = statistics.contrast_cross_covariance
        changed_cross = original_cross.copy()
        changed_cross[0, 0] += 1.0e-12
        changed_cross.setflags(write=False)
        object.__setattr__(
            statistics, "contrast_cross_covariance", changed_cross
        )
        try:
            with self.assertRaisesRegex(
                rad.RademacherAdapterError, "cross covariance.*relabelled"
            ):
                rad.fit_rademacher_fold(
                    self.cells,
                    self.moments,
                    self.candidates,
                    self.geometry,
                    self.features,
                    self.physical_prepared,
                    self.physical_pairs,
                )
        finally:
            object.__setattr__(
                statistics, "contrast_cross_covariance", original_cross
            )

        replacement = m2.PairUniformStatistics(
            mean=statistics.mean,
            covariance=statistics.covariance,
            contrast_cross_covariance=statistics.contrast_cross_covariance,
        )
        original_statistics = self.features._statistics[8]
        self.features._statistics[8] = replacement
        try:
            with self.assertRaisesRegex(
                rad.RademacherAdapterError, "statistics cache was relabelled"
            ):
                rad.fit_rademacher_fold(
                    self.cells,
                    self.moments,
                    self.candidates,
                    self.geometry,
                    self.features,
                    self.physical_prepared,
                    self.physical_pairs,
                )
        finally:
            self.features._statistics[8] = original_statistics

    def test_candidate_basis_mutation_rejected_at_beta_zero_and_signed_zero(
        self,
    ) -> None:
        zero = np.zeros(rad.COLUMN_COUNT, dtype=float)
        basis = self.candidates.basis(4)
        original_contrast_object = basis.contrasts
        original_contrasts = original_contrast_object.copy()
        basis.contrasts[0, 0] += 1.0e-12
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2.evaluate_m2(
                    zero,
                    self.fixed_m1_theta,
                    self.validation_prepared,
                    self.candidates,
                    self.geometry,
                    self.features,
                )
        finally:
            basis.contrasts[...] = original_contrasts

        replacement = basis.contrasts.copy()
        object.__setattr__(basis, "contrasts", replacement)
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2.candidate_distribution(
                    zero,
                    self.fixed_m1_theta,
                    self.validation_prepared,
                    4,
                    self.candidates,
                    self.geometry,
                    self.features,
                )
        finally:
            object.__setattr__(basis, "contrasts", original_contrast_object)

        zero_basis = self.candidates.basis(0)
        original_zero_contrasts = zero_basis.contrasts.copy()
        zero_basis.contrasts[0, 0] = -0.0
        self.assertTrue(zero_basis.contrasts[0, 0] == 0.0)
        self.assertTrue(np.signbit(zero_basis.contrasts[0, 0]))
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2.evaluate_m2(
                    zero,
                    self.fixed_m1_theta,
                    self.fitted.prepared,
                    self.candidates,
                    self.geometry,
                    self.features,
                )
        finally:
            zero_basis.contrasts[...] = original_zero_contrasts

    def test_physical_inputs_and_normalization_mutations_fail_closed(self) -> None:
        rows = np.ones((1, rad.COLUMN_COUNT), dtype=float)

        original_projector = self.physical_transform.range_projector
        object.__setattr__(
            self.physical_transform,
            "range_projector",
            np.zeros_like(original_projector),
        )
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2._range_project_rows(rows, self.fitted.transform)
        finally:
            object.__setattr__(
                self.physical_transform, "range_projector", original_projector
            )

        original_center = float(
            self.physical_prepared.m1_data.geometry_centers[0, 0]
        )
        self.physical_prepared.m1_data.geometry_centers[0, 0] += 0.25
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2._range_project_rows(rows, self.fitted.transform)
        finally:
            self.physical_prepared.m1_data.geometry_centers[0, 0] = original_center

        original_distance = float(self.cells[0].soma_distance_mm[0])
        self.cells[0].soma_distance_mm[0] += 0.5
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2._range_project_rows(rows, self.fitted.transform)
        finally:
            self.cells[0].soma_distance_mm[0] = original_distance

        original_covariance = float(self.moments[0].raw_joint_covariance[0, 0])
        self.moments[0].raw_joint_covariance[0, 0] += 0.5
        try:
            with self.assertRaisesRegex(m2.M2Error, "authentication"):
                m2._range_project_rows(rows, self.fitted.transform)
        finally:
            self.moments[0].raw_joint_covariance[0, 0] = original_covariance

    def test_validation_is_bound_to_its_exact_dictionary_even_at_beta_zero(self) -> None:
        wrong = rad.RademacherFeatureLibrary(
            self.candidates,
            rad.RademacherSeedRoute(
                "target_free_qualification", DIRECT_SEED + 123
            ),
        )
        zero = np.zeros(rad.COLUMN_COUNT, dtype=float)
        with self.assertRaisesRegex(m2.M2Error, "dictionary"):
            m2.evaluate_m2(
                zero,
                self.fixed_m1_theta,
                self.validation_prepared,
                self.candidates,
                self.geometry,
                wrong,
            )
        with self.assertRaisesRegex(m2.M2Error, "dictionary"):
            m2.residual_pair_batch(
                self.validation_prepared,
                4,
                0,
                8,
                self.candidates,
                self.geometry,
                wrong,
            )

    def test_validation_only_k_is_anchored_before_apply(self) -> None:
        training_indexes = tuple(
            index for index, cell in enumerate(self.cells) if cell.k != 4
        )
        self.assertTrue(training_indexes)
        self.assertNotIn(4, {self.cells[index].k for index in training_indexes})
        _, physical_prepared = m2.fit_residual_transform(
            self.cells,
            self.moments,
            self.candidates,
            self.geometry,
            self.physical_pairs,
            indexes=training_indexes,
            expected_rank=rad.EXPECTED_PROJECTED_RANK,
        )
        fitted = rad.fit_rademacher_fold(
            self.cells,
            self.moments,
            self.candidates,
            self.geometry,
            self.features,
            physical_prepared,
            self.physical_pairs,
            indexes=training_indexes,
            expected_rank=rad.EXPECTED_PROJECTED_RANK,
        )
        validation_index = next(
            index for index, cell in enumerate(self.cells) if cell.k == 4
        )
        validation_cells = (self.cells[validation_index],)
        validation_moments = (self.moments[validation_index],)
        basis = self.candidates.basis(4)
        original_object = basis.contrasts
        original_values = original_object.copy()

        basis.contrasts[0, 0] += 1.0e-12
        try:
            with self.assertRaisesRegex(
                rad.RademacherAdapterError, "physical fold provenance"
            ):
                rad.apply_rademacher_validation(
                    fitted, validation_cells, validation_moments
                )
        finally:
            basis.contrasts[...] = original_values

        replacement = original_object.copy()
        object.__setattr__(basis, "contrasts", replacement)
        try:
            with self.assertRaisesRegex(
                rad.RademacherAdapterError, "physical fold provenance"
            ):
                rad.apply_rademacher_validation(
                    fitted, validation_cells, validation_moments
                )
        finally:
            object.__setattr__(basis, "contrasts", original_object)

        zero_locations = np.argwhere(
            (original_object == 0.0) & ~np.signbit(original_object)
        )
        self.assertGreater(len(zero_locations), 0)
        row, column = (int(value) for value in zero_locations[0])
        original_object[row, column] = -0.0
        self.assertTrue(np.signbit(original_object[row, column]))
        try:
            with self.assertRaisesRegex(
                rad.RademacherAdapterError, "physical fold provenance"
            ):
                rad.apply_rademacher_validation(
                    fitted, validation_cells, validation_moments
                )
        finally:
            original_object[...] = original_values

    def test_mismatched_physical_training_fold_fails_closed(self) -> None:
        wrong_design = m1.DesignTransform(
            self.physical_transform.design_transform.soma_mean + 1.0,
            self.physical_transform.design_transform.soma_scale,
        )
        mismatched = dataclasses.replace(
            self.physical_transform,
            design_transform=wrong_design,
        )
        with self.assertRaisesRegex(
            rad.RademacherAdapterError,
            "same-pass provenance",
        ):
            rad.fit_rademacher_fold(
                self.cells,
                self.moments,
                self.candidates,
                self.geometry,
                self.features,
                dataclasses.replace(
                    self.physical_prepared,
                    transform=mismatched,
                ),
                self.physical_pairs,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
