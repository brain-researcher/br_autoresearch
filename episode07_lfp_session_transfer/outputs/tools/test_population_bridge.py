"""Independent bridge arithmetic/support fixtures; execute only in Slurm.

These small integer-count constructions satisfy the actual complete-role and
native-window rules. They test implementation, not scientific recovery or model
qualification. No capacity-matched history/today predictor is asserted here.
"""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import numpy as np

from native_trials import BANDS, DayTrials, EARLIER
import population_bridge as core


def check():
    pass


def latent_cells(*, unsupported=False):
    # Four exactly centered, mutually orthogonal columns over all 8 directions
    # and 10 native M1 bins. The fourth normally varies in native time.
    rows = []
    for direction in range(8):
        for time_bin in range(10):
            signs = [1 if not (direction & bit) else -1 for bit in (1, 2, 4)]
            fourth = signs[0] * signs[1] if unsupported else (1 if time_bin % 2 == 0 else -1)
            rows.append(signs + [fourth])
    return np.asarray(rows, dtype=np.float64)


def native_day(index, loading, *, unsupported=False):
    dates = EARLIER["Mihili"] + ("20140304",)
    repetitions = 3 if index == 3 else 15
    directions = np.repeat(np.arange(8, dtype=np.int64), repetitions)
    anchors = 100 + 20 * index + latent_cells(unsupported=unsupported) @ loading
    spikes = anchors.reshape(8, 10, loading.shape[1])[directions]
    guide = np.asarray([(7 + index, lo, hi) for lo, hi in BANDS], dtype=np.float64)
    # Repeated numeric unit labels across days must not cause an intersection
    # or a shared native axis. In these fixtures day widths differ explicitly.
    units = np.asarray([(unit + 1, 1) for unit in range(loading.shape[1])], dtype=np.float64)
    return DayTrials(animal="Mihili", day=dates[index], target_day=dates[-1],
        region="M1", stream="history_assisted", view_role="fit",
        role="calibration" if index == 3 else "training",
        trial_ids=tuple(f"trial_{number + 1:06d}" for number in range(len(directions))),
        directions=directions, bin_offsets=np.arange(5, 15, dtype=np.int64),
        lfp=np.zeros((len(directions), 10, 9), dtype=np.float32),
        lfp_guide=guide, spikes=spikes, unit_guide=units)


def full_roster(*, unsupported_source=None):
    extra = np.asarray([[1, 0, 1], [0, 1, -1], [1, 0, 0], [0, 1, 1]])
    loadings, days = [], []
    for index in range(4):
        loading = np.column_stack((np.diag([12 + index, 6, 3, 1]), extra[:, :index]))
        loadings.append(loading)
        days.append(native_day(index, loading, unsupported=index == unsupported_source))
    return tuple(days), tuple(loadings)


def literal_anchors(day):
    result = np.empty((80, day.spikes.shape[-1]), dtype=np.float64)
    for direction in range(8):
        trials = [i for i, value in enumerate(day.directions) if value == direction]
        for time_bin in range(10):
            for neuron in range(day.spikes.shape[-1]):
                result[10 * direction + time_bin, neuron] = math.fsum(
                    float(day.spikes[trial, time_bin, neuron]) for trial in trials) / len(trials)
    return result


class PopulationBridgeTests(unittest.TestCase):
    def test_rank_fallback_preserves_native_axes_and_unresidualized_time_anchors(self):
        days, loadings = full_roster()
        bridge = core.fit_history_bridge(days, alignment="orthogonal", check=check)
        self.assertEqual(bridge.rank, 4)
        self.assertEqual(tuple(rank for rank, _ in bridge.rejected_ranks), (16, 12, 8))
        self.assertTrue(all(reason == "fitting_population_rank_below_candidate"
                            for _, reason in bridge.rejected_ranks))
        bases = tuple(link.source for link in bridge.sources) + (bridge.target,)
        self.assertEqual(tuple(basis.components.shape for basis in bases),
                         ((4, 4), (5, 4), (6, 4), (7, 4)))
        for day_index, (day, loading, basis) in enumerate(zip(days, loadings, bases)):
            np.testing.assert_array_equal(basis.unit_guide, day.unit_guide)
            np.testing.assert_array_equal(basis.bin_offsets, np.arange(5, 15))
            self.assertEqual(basis.trial_ids, day.trial_ids)
            raw = literal_anchors(day)
            np.testing.assert_array_equal(raw, 100 + 20 * day_index
                                          + latent_cells() @ loading)
            projected = (raw - basis.mean) @ basis.components
            projected -= np.mean(projected, axis=0)
            np.testing.assert_allclose(basis.centered_anchors, projected, rtol=0, atol=1e-11)
            # Regressing out direction/time before fitting would leave zero
            # responses in this construction and could not pass rank four.
            self.assertGreater(np.linalg.norm(basis.centered_anchors), 1)
            self.assertGreater(np.linalg.norm(projected[0] - projected[1]), 1)
            expected_space = np.linalg.qr(loading.T, mode="reduced")[0]
            np.testing.assert_allclose(basis.components @ basis.components.T,
                                      expected_space @ expected_space.T, rtol=0, atol=1e-11)
            self.assertLessEqual(basis.anchor_condition, 30)
            with self.assertRaises(ValueError):
                basis.components.setflags(write=True)
        # The raw cross-covariance is strongly anisotropic but both anchor
        # column spaces agree. The specified QR gate must give ratio one.
        cross = bases[0].centered_anchors.T @ bases[-1].centered_anchors
        raw_singular = np.linalg.svd(cross, compute_uv=False)
        self.assertLess(raw_singular[-1] / raw_singular[0], 0.05)
        for link in bridge.sources:
            self.assertAlmostEqual(link.support_ratio, 1.0, places=12)

    def test_one_unsupported_source_invalidates_all_sources_without_omission(self):
        days, _ = full_roster(unsupported_source=1)
        # Every day still has four identified within-day directions/time axes;
        # the one source's fourth axis is orthogonal to the target's fourth.
        target = latent_cells()
        unsupported = latent_cells(unsupported=True)
        normalized_cross = unsupported.T @ target / 80
        np.testing.assert_array_equal(normalized_cross, np.diag([1., 1., 1., 0.]))
        with self.assertRaisesRegex(core.BridgeUnavailable, "source_target_Q_support_ratio_below_0p05"):
            core.fit_history_bridge(days, alignment="orthogonal", check=check)
        with self.assertRaisesRegex(ValueError, "exactly three source days"):
            core.fit_history_bridge((days[0], days[2], days[3]), alignment="orthogonal", check=check)

    def test_transport_matches_analytic_orthogonal_and_regularized_maps(self):
        scales = (np.asarray([3., 12., 1., 6.]), np.asarray([9., 2., 5., 1.]),
                  np.asarray([2., 7., 1., 4.]), np.asarray([10., 8., 4., 2.]))
        permutations = ((2, 0, 3, 1), (1, 3, 0, 2), (3, 1, 2, 0), (1, 0, 3, 2))
        signs = ((-1, 1, 1, -1), (1, -1, 1, -1), (-1, -1, 1, 1), (1, -1, -1, 1))
        days, orders, native_signs = [], [], []
        for index in range(4):
            perm = np.asarray(permutations[index])
            loading = np.diag(scales[index])[:, perm] * np.asarray(signs[index])[None, :]
            days.append(native_day(index, loading))
            orders.append(np.argsort(-scales[index]))
            native_signs.append({int(latent): signs[index][int(np.flatnonzero(perm == latent)[0])]
                                 for latent in range(4)})
        ridge = 7.5
        for alignment in ("none", "orthogonal", "regularized_correlation"):
            bridge = core.fit_history_bridge(tuple(days), alignment=alignment,
                regularization=ridge if alignment == "regularized_correlation" else None, check=check)
            self.assertEqual(bridge.rank, 4)
            for source_index, link in enumerate(bridge.sources):
                expected = np.zeros((4, 4))
                for source_column, latent in enumerate(orders[source_index]):
                    target_column = int(np.flatnonzero(orders[3] == latent)[0])
                    value = native_signs[source_index][int(latent)] * native_signs[3][int(latent)]
                    if alignment == "regularized_correlation":
                        # Each balanced latent has sum of squares 80. This
                        # independently pins the sample covariance denominator.
                        source_variance = 80 / 79 * scales[source_index][latent] ** 2
                        target_variance = 80 / 79 * scales[3][latent] ** 2
                        value *= math.sqrt((target_variance + ridge) / (source_variance + ridge))
                    expected[source_column, target_column] = value
                if alignment == "none":
                    expected = np.eye(4)
                np.testing.assert_allclose(link.matrix, expected, rtol=0, atol=2e-11)

    def test_today_only_scope_and_numerical_or_cancellation_failures_do_not_fallback(self):
        days, _ = full_roster()
        today = replace(days[-1], stream="today_only")
        basis = core.fit_today_calibration_basis(today, rank=4, check=check)
        self.assertEqual(basis.stream, "today_only")
        self.assertEqual(basis.trial_ids, today.trial_ids)
        np.testing.assert_array_equal(basis.unit_guide, today.unit_guide)

        class ForbiddenArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError("forbidden history state reached numerical access")

        history_bound = replace(days[-1], spikes=ForbiddenArray())
        with self.assertRaises(PermissionError):
            core.fit_today_calibration_basis(history_bound, rank=4, check=check)
        with self.assertRaises(PermissionError):
            core.fit_history_bridge(days[:-1] + (replace(today, spikes=ForbiddenArray()),),
                                    alignment="orthogonal", check=check)
        with self.assertRaises(PermissionError):
            core.fit_history_bridge((days[1], days[0], days[2], days[3]),
                                    alignment="orthogonal", check=check)
        with patch.object(core.np.linalg, "svd", side_effect=np.linalg.LinAlgError("fixture decomposition")) as svd:
            with self.assertRaisesRegex(np.linalg.LinAlgError, "fixture decomposition"):
                core.fit_history_bridge(days, alignment="orthogonal", check=check)
            self.assertEqual(svd.call_count, 1)
        def cancel():
            raise RuntimeError("fixture cancelled")
        with self.assertRaisesRegex(RuntimeError, "fixture cancelled"):
            core.fit_history_bridge(days, alignment="orthogonal", check=cancel)


if __name__ == "__main__":
    unittest.main()
