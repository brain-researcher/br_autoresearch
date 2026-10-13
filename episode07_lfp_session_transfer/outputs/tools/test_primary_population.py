"""Focused primary-coordinate fixtures; execute only in coordinated Slurm.

These fabricated native counts check implementation, not empirical support or
scientific recovery. Residual-bridge eligibility is deliberately not imported
into primary model fitting. No test/import was executed while writing this.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

import primary_population as core
from test_population_bridge import full_roster, literal_anchors, native_day


def check():
    pass


class ForbiddenArray:
    def __array__(self, *args, **kwargs):
        raise AssertionError("refused scope reached response-array access")


class PrimaryPopulationTests(unittest.TestCase):
    def test_primary_uses_full_native_axes_without_conditional_bridge_gates(self):
        days, _ = full_roster(unsupported_source=1)
        # One source has condition 40; another has an orthogonal fourth anchor
        # dimension. Both violate explanatory gates, yet define primary fits.
        days = (native_day(0, np.diag([40., 6., 3., 1.])),) + days[1:]
        for alignment in ("none", "orthogonal", "regularized_correlation"):
            fitted = core.fit_primary_population(days, rank=4, alignment=alignment,
                regularization=3. if alignment == "regularized_correlation" else None,
                check=check)
            self.assertEqual(fitted.rank, 4)
            self.assertEqual(tuple(basis.components.shape for basis in fitted.bases),
                             ((4, 4), (5, 4), (6, 4), (7, 4)))
            self.assertEqual(len(fitted.source_maps), 3)
            for day, basis in zip(days, fitted.bases):
                self.assertEqual(basis.trial_ids, day.trial_ids)
                np.testing.assert_array_equal(basis.unit_guide, day.unit_guide)
                np.testing.assert_array_equal(basis.bin_offsets, np.arange(5, 15))
                self.assertEqual(basis.centered_anchors.shape, (80, 4))
                expected = (literal_anchors(day) - basis.mean) @ basis.components
                expected -= expected.mean(axis=0)
                np.testing.assert_allclose(basis.centered_anchors, expected,
                                           rtol=0, atol=1e-11)
                with self.assertRaises(ValueError):
                    basis.components.setflags(write=True)
            self.assertGreater(np.linalg.cond(fitted.bases[0].centered_anchors), 30.)
            source_q = np.linalg.qr(fitted.bases[1].centered_anchors, mode="reduced")[0]
            target_q = np.linalg.qr(fitted.bases[-1].centered_anchors, mode="reduced")[0]
            singular = np.linalg.svd(source_q.T @ target_q, compute_uv=False)
            self.assertLess(singular[-1] / singular[0], .05)
            for matrix in fitted.source_maps:
                self.assertEqual(matrix.shape, (4, 4))
                self.assertTrue(np.isfinite(matrix).all())
                if alignment == "none":
                    np.testing.assert_array_equal(matrix, np.eye(4))
                elif alignment == "orthogonal":
                    np.testing.assert_allclose(matrix.T @ matrix, np.eye(4),
                                               rtol=0, atol=1e-12)

    def test_explicit_rank_has_no_fallback_or_numerical_retry(self):
        days, _ = full_roster()
        # These days support rank four. A silent 8 -> 4 fallback would return
        # a fit and fail this test instead of reporting the requested failure.
        with self.assertRaisesRegex(core.PrimaryPopulationUnavailable,
                                    "below_explicit_candidate_rank"):
            core.fit_primary_population(days, rank=8, alignment="none", check=check)
        unread = tuple(replace(day, spikes=ForbiddenArray()) for day in days)
        for rank in (True, 4., 0, 5):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                core.fit_primary_population(unread, rank=rank, alignment="none", check=check)
        with patch.object(core.np.linalg, "svd",
                          side_effect=np.linalg.LinAlgError("fixture decomposition")) as svd:
            with self.assertRaisesRegex(np.linalg.LinAlgError, "fixture decomposition"):
                core.fit_primary_population(days, rank=4, alignment="orthogonal", check=check)
            self.assertEqual(svd.call_count, 1)
        def cancel():
            raise RuntimeError("fixture cancellation")
        with self.assertRaisesRegex(RuntimeError, "fixture cancellation"):
            core.fit_primary_population(unread, rank=4, alignment="none", check=cancel)

    def test_projection_and_reconstruction_keep_native_axes_and_residual_origin(self):
        days, loadings = full_roster()
        target = days[-1]
        # Real within-direction trial variation in an existing population
        # direction keeps rank four and makes residual-origin errors visible.
        residuals = ((np.arange(len(target.trial_ids)) % 3 - 1)[:, None, None]
                     * loadings[-1][0][None, None, :])
        residuals = np.broadcast_to(residuals, target.spikes.shape).copy()
        target = replace(target, spikes=target.spikes + residuals)
        days = days[:-1] + (target,)
        fitted = core.fit_primary_population(days, rank=4, alignment="orthogonal", check=check)
        basis = fitted.bases[-1]
        coordinates = core.project_primary_responses(fitted, target, target.spikes,
                                                    residualized=False, check=check)
        restored = core.reconstruct_primary_responses(fitted, coordinates,
                                                      residualized=False, check=check)
        np.testing.assert_allclose(restored, target.spikes, rtol=0, atol=2e-11)
        residual_coordinates = core.project_primary_responses(fitted, target, residuals,
                                                             residualized=True, check=check)
        residual_restored = core.reconstruct_primary_responses(fitted, residual_coordinates,
                                                               residualized=True, check=check)
        np.testing.assert_allclose(residual_restored, residuals, rtol=0, atol=2e-11)
        zeros = np.zeros((2, 10, 4))
        np.testing.assert_array_equal(core.reconstruct_primary_responses(
            fitted, zeros, residualized=True, check=check), np.zeros((2, 10, 7)))
        np.testing.assert_array_equal(core.reconstruct_primary_responses(
            fitted, zeros, residualized=False, check=check),
            np.broadcast_to(basis.mean, (2, 10, 7)))
        # Independent scalar projection checks the source's own neuron width
        # and right-side transport, rather than pooling unrelated native axes.
        source = days[1]
        source_basis, mapping = fitted.bases[1], fitted.source_maps[1]
        source_projection = core.project_primary_responses(fitted, source, source.spikes,
                                                           residualized=False, check=check)
        expected_row = []
        centered = source.spikes[0, 0] - source_basis.mean
        native_coordinates = [math.fsum(float(centered[n]) * float(source_basis.components[n, j])
                                       for n in range(len(centered))) for j in range(4)]
        for j in range(4):
            expected_row.append(math.fsum(native_coordinates[k] * float(mapping[k, j])
                                          for k in range(4)))
        np.testing.assert_allclose(source_projection[0, 0], expected_row, rtol=0, atol=1e-11)
        extreme = np.broadcast_to(-100000. * basis.components[0], (2, 10, 4))
        negative = core.reconstruct_primary_responses(fitted, extreme,
                                                      residualized=False, check=check)
        self.assertTrue(np.all(negative[:, :, 0] < 0))  # no prediction clipping
        for value in (coordinates, residual_restored, source_projection):
            with self.assertRaises(ValueError):
                value.setflags(write=True)

    def test_stream_roles_and_trial_native_axis_bindings_refuse_before_responses(self):
        days, _ = full_roster()
        history = core.fit_primary_population(days, rank=4, alignment="none", check=check)
        today = replace(days[-1], stream="today_only")
        fitted = core.fit_primary_population((today,), rank=4, alignment="none", check=check)
        self.assertEqual(fitted.stream, "today_only")
        self.assertEqual(fitted.source_maps, ())
        np.testing.assert_allclose(fitted.bases[0].components,
                                   history.bases[-1].components, rtol=0, atol=1e-12)
        self.assertEqual(fitted.bases[0].trial_ids, today.trial_ids)
        for roster in (days[:-1] + (replace(today, spikes=ForbiddenArray()),),
                       days[1:], (replace(days[-1], spikes=ForbiddenArray()),),
                       (replace(today, view_role="development_features",
                                role="development", spikes=ForbiddenArray()),)):
            with self.assertRaises(PermissionError):
                core.fit_primary_population(roster, rank=4, alignment="none", check=check)
        with self.assertRaises(ValueError):
            core.fit_primary_population((replace(today, spikes=ForbiddenArray()),),
                                        rank=4, alignment="orthogonal", check=check)
        invalid = (days[-1], replace(today, trial_ids=today.trial_ids[::-1]),
                   replace(today, unit_guide=today.unit_guide[::-1]),
                   replace(today, bin_offsets=np.arange(6, 16)))
        for record in invalid:
            with self.assertRaises(PermissionError):
                core.project_primary_responses(fitted, record, ForbiddenArray(),
                                               residualized=False, check=check)
        with self.assertRaises(PermissionError):
            core.project_primary_responses(history, today, ForbiddenArray(),
                                           residualized=False, check=check)
        with self.assertRaises(TypeError):
            core.project_primary_responses(fitted, today, ForbiddenArray(),
                                           residualized=1, check=check)
        with self.assertRaises(ValueError):
            core.reconstruct_primary_responses(fitted, np.zeros((2, 9, 4)),
                                               residualized=False, check=check)


if __name__ == "__main__":
    unittest.main()
