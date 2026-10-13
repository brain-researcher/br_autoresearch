"""Seconds-scale equivalence checks for cached registered-null arithmetic only."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import native_rsa as native
import run_registered_null_unit as nulls


class RegisteredNullUnitTests(unittest.TestCase):
    def test_cached_svd_matches_original_tall_wide_and_rank_deficient_geometry(self):
        rng = np.random.Generator(np.random.PCG64(917))
        tall = rng.normal(size=(9, 4)) + np.arange(4)
        wide = rng.normal(size=(6, 9)) + np.arange(9)
        rank_deficient = np.column_stack((tall, 2*tall[:, 0], tall[:, 1]+tall[:, 2]))
        for value in (tall, wide, rank_deficient):
            original = value.copy()
            factors = nulls.factor_geometry(np, native, value)
            for model_index, draw_index in ((0, 0), (1, 7), (2, 999)):
                with self.subTest(shape=value.shape, model=model_index, draw=draw_index):
                    raw, normalized = nulls.geometry_from_factors(
                        np, native, factors, model_index, draw_index)
                    expected_raw, expected_normalized = native.covariance_matched_geometry(
                        value, model_index, draw_index)
                    np.testing.assert_array_equal(raw, expected_raw)
                    np.testing.assert_array_equal(normalized, expected_normalized)
                    np.testing.assert_allclose(raw.mean(axis=0), value.mean(axis=0), atol=1e-12)
                    centered = value-value.mean(axis=0)
                    generated = raw-raw.mean(axis=0)
                    np.testing.assert_allclose(generated.T @ generated, centered.T @ centered,
                                               rtol=1e-11, atol=1e-11)
                    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1., atol=1e-12)
            np.testing.assert_array_equal(value, original)

    def test_cached_rank_permutation_matches_full_tied_reranking_with_shared_bijection(self):
        rng = np.random.Generator(np.random.PCG64(230))
        size = 9
        edges = np.triu_indices(size, 1)

        def tied_matrix():
            matrix = np.zeros((size, size), dtype=float)
            matrix[edges] = rng.integers(0, 8, size=len(edges[0]))
            return matrix+matrix.T

        models = [tied_matrix() for _ in range(4)]
        neural = [tied_matrix() for _ in range(3)]
        originals = [item.copy() for item in models+neural]
        strata = np.array(['laion', 'things', 'thingsplus']*3)
        cache = nulls.rank_cache(np, native, models, neural)
        for draw in (0, 7, 255, 9999):
            permutation = native.whole_image_permutation(strata, draw)
            np.testing.assert_array_equal(strata[permutation], strata)
            with patch.object(native, 'whole_image_permutation',
                              wraps=native.whole_image_permutation) as shared:
                actual_r, actual_z = nulls.permutation_scores(np, native, cache, strata, draw)
                shared.assert_called_once()
            expected_r = np.array([[native.spearman_rsa(native.relabel_rdm(model, permutation), target)
                                    for target in neural] for model in models])
            expected_z = np.array([[native.fisher_z(float(value)) for value in row] for row in expected_r])
            np.testing.assert_allclose(actual_r, expected_r, rtol=0, atol=1e-14)
            np.testing.assert_allclose(actual_z, expected_z, rtol=0, atol=1e-14)
        for actual, original in zip(models+neural, originals):
            np.testing.assert_array_equal(actual, original)

    def test_signed_null_scores_survive_and_undefined_or_asymmetric_inputs_fail(self):
        edges = np.triu_indices(4, 1)
        model = np.zeros((4, 4))
        neural = np.zeros((4, 4))
        model[edges] = [1., 2., 3., 4., 5., 6.]
        neural[edges] = [6., 5., 4., 3., 1., 2.]
        model += model.T
        neural += neural.T
        cache = nulls.rank_cache(np, native, [model], [neural])
        # Singleton strata force the identity relabeling; no draw-dependent fixture.
        actual_r, actual_z = nulls.permutation_scores(np, native, cache, ['a', 'b', 'c', 'd'], 0)
        self.assertLess(actual_r[0, 0], 0)
        self.assertLess(actual_z[0, 0], 0)
        self.assertAlmostEqual(actual_r[0, 0], native.spearman_rsa(model, neural), places=14)
        asymmetric = model.copy()
        asymmetric[0, 1] += .5
        with self.assertRaises(ValueError):
            nulls.rank_cache(np, native, [asymmetric], [neural])
        with self.assertRaises(native.NotEvaluable):
            nulls.rank_cache(np, native, [np.zeros_like(model)], [neural])
    def test_registered_ranges_and_original_clock_are_bounded_without_reset(self):
        self.assertEqual(nulls.draw_range('0:256', 10000), (0, 256))
        self.assertEqual(nulls.draw_range('256:10000', 10000), (256, 10000))
        self.assertEqual(nulls.draw_range('8:1000', 1000), (8, 1000))
        for text in ('-1:8', '8:7', '0:1001', '0:8:9', '0: 8'):
            with self.assertRaises(ValueError):
                nulls.draw_range(text, 1000)
        first = datetime(2026, 10, 5, 22, tzinfo=timezone.utc)
        now = first+timedelta(hours=119, minutes=59)
        deadline, end = nulls.clock_deadline(first.isoformat(), now, 100., 1300.)
        self.assertEqual(deadline, 160.)
        self.assertEqual(datetime.fromisoformat(end), first+timedelta(hours=120))
        self.assertEqual(nulls.clock_deadline(first.isoformat(), first, 100., 1300.)[0], 1300.)
        with self.assertRaises(TimeoutError):
            nulls.clock_deadline(first.isoformat(), first+timedelta(hours=120), 100., 1300.)
        with self.assertRaises(ValueError):
            nulls.clock_deadline(first.replace(tzinfo=None).isoformat(), now, 100., 1300.)


if __name__ == '__main__':
    unittest.main()
