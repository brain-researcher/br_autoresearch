"""Three array-only partition checks; run only in coordinated Slurm compute.

Existing metric/batched tests supply the numerical scorer coverage. These
fixtures exercise fixed partition dispatch, exact joins and failed work, with
all 1,000 draws and all original slots retained; there is no recovery study.
"""
from collections import Counter
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import sys
from types import MappingProxyType
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import finding_a_draw_batches as partition
import finding_a_batched_inference as batched
import finding_a_inference as original
from test_finding_a_batched_inference import equal, percentile
from test_finding_a_inference import TEST_IDS, TRAIN_IDS, fixture_panel, readonly


def fixture_factory(schedule):
    """Explicit compact summaries, reduced by the existing panel reducer.

    Means depend on the number of sampled occurrences of image zero. A single
    profile contrast draw is unavailable; one within-source point is missing
    even though all of its draws are finite. Neither is rescued by the join.
    """
    template = fixture_panel()

    @lru_cache(maxsize=64)
    def fields(occurrences, bad_contrast, point):
        values = []
        for metric_name in original._METRICS:
            base = getattr(template, metric_name)
            means = {key: value.copy() + occurrences / 1024 for key, value in base.layer_means.items()}
            reasons = {key: value.copy() for key, value in base.layer_reasons.items()}
            for key in original.ENDPOINTS[1:]:
                reasons[key][:, np.arange(5), np.arange(5)] = "self_pair_not_applicable"
            if point:
                means["within_source"][:, 4] = np.nan
                reasons["within_source"][:, 4] = "fixture_point_unavailable"
            if bad_contrast and metric_name == "profile":
                means["content_minus_brain"][3, 1, 4] = np.nan
                reasons["content_minus_brain"][3, 1, 4] = "fixture_support_mismatch"
            values.append(original._reduce((means, base.layer_defined_counts,
                base.layer_undefined_counts, reasons), lambda: None))
        return tuple(values)

    def panel(indices, *, point=False):
        pattern, profile = fields(int(np.count_nonzero(indices == 0)),
                                  np.array_equal(indices, schedule[37]), point)
        return original.PanelResult(TEST_IDS, readonly(indices, np.int64),
                                    original.PARTICIPANTS, original.LAYERS, pattern, profile)

    return panel


class FindingADrawBatchTests(unittest.TestCase):
    def test_join_matches_original_bootstrap_and_fixed_pointwise_failure_rules(self):
        self.assertEqual((partition.DRAW_BATCH_SIZE, partition.BATCH_COUNT), (16, 63))
        self.assertEqual((original.BOOTSTRAP_DRAWS, original.BOOTSTRAP_SEED), (1000, 42))
        schedule = readonly(np.random.RandomState(42).randint(0, 224, size=(1000, 224)), np.int64)
        make_panel = fixture_factory(schedule)
        point = make_panel(np.arange(224), point=True)
        requests = []

        def supplied(provider, training, images, indices, *, check):
            check()
            self.assertEqual(dict(training), TRAIN_IDS)
            self.assertEqual(images, TEST_IDS)
            self.assertFalse(indices.flags.writeable)
            requests.append(indices.copy())
            return tuple(make_panel(draw) for draw in indices)

        before = np.random.get_state()
        closed = []
        with mock.patch.object(batched, "_evaluate_panel_batch", side_effect=supplied), \
                mock.patch.object(original, "evaluate_panel", return_value=point) as point_evaluator:
            expected = batched.bootstrap_panel(object(), TRAIN_IDS, TEST_IDS, check=lambda: None)

            def completed():
                try:
                    # Task completion order must not alter canonical draw order.
                    for index in range(62, -1, -1):
                        yield partition.evaluate_draw_batch(object(), TRAIN_IDS, TEST_IDS,
                                                            batch_index=index, check=lambda: None)
                finally:
                    closed.append(True)

            actual = partition.assemble_bootstrap(point, completed(), check=lambda: None)
            self.assertEqual(point_evaluator.call_count, 1)  # join never recomputes the point
        equal(self, actual, expected)
        self.assertIs(actual.point, point)
        self.assertEqual(closed, [True])
        self.assertEqual(len(requests), 126)
        np.testing.assert_array_equal(np.concatenate(requests[:63]), schedule)
        np.testing.assert_array_equal(np.concatenate(requests[63:][::-1]), schedule)
        self.assertEqual(Counter(map(len, requests)), {16: 124, 8: 2})
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertFalse(actual.draw_indices.flags.writeable)
        failed = actual.profile.layer_dyad["content_minus_brain"]
        self.assertFalse(failed.available[3, 1, 4])
        self.assertEqual(failed.valid_draws[3, 1, 4], 999)
        self.assertTrue(np.isnan(failed.lower[3, 1, 4]))
        self.assertTrue(actual.pattern.layer_dyad["content_minus_brain"].available[3, 1, 4])
        point_missing = actual.pattern.source["within_source"]
        self.assertFalse(point_missing.available[4])
        self.assertEqual(point_missing.valid_draws[4], 1000)
        values = [make_panel(row).pattern.source_means["within_source"][0] for row in schedule]
        self.assertAlmostEqual(point_missing.lower[0], percentile(values, .025), places=14)
        self.assertAlmostEqual(point_missing.upper[0], percentile(values, .975), places=14)

    def test_boundary_and_tail_batches_reuse_full_original_panel_dispatch(self):
        schedule = readonly(np.random.RandomState(42).randint(0, 224, size=(1000, 224)), np.int64)
        expected_slots = {(method, source, target, layer) for layer in original.LAYERS
            for source in original.PARTICIPANTS for method, target in [("within_source", None)] + [
                (method, target) for target in original.PARTICIPANTS if source != target
                for method in ("content_loss", "brain_loss")]}
        self.assertEqual(len(expected_slots), 855)
        for index, count in ((0, 16), (61, 16), (62, 8)):
            requests = []
            training = dict(TRAIN_IDS)

            def provider(*, method, source, target, layer):
                training["sub-03"] = ("caller mapping changed after snapshot",)
                requests.append((method, source, target, layer))
                return original.UnavailableLayer("explicit_fixture_missing")

            with mock.patch.object(batched, "score_layer_batch", side_effect=AssertionError("missing input scored")):
                result = partition.evaluate_draw_batch(provider, training, TEST_IDS,
                                                       batch_index=index, check=lambda: None)
            self.assertEqual(result.batch_index, index)
            self.assertEqual(len(result.panels), count)
            self.assertEqual(set(requests), expected_slots)
            self.assertEqual(len(requests), 855)
            self.assertEqual(Counter(row[0] for row in requests),
                             {"within_source": 95, "content_loss": 380, "brain_loss": 380})
            for offset, panel in enumerate(result.panels):
                self.assertEqual(panel.image_ids, TEST_IDS)
                self.assertEqual(panel.layers, original.LAYERS)
                self.assertEqual(panel.participants, original.PARTICIPANTS)
                np.testing.assert_array_equal(panel.image_indices, schedule[index * 16 + offset])
                self.assertFalse(panel.image_indices.flags.writeable)
                for name in original._METRICS:
                    metric = getattr(panel, name)
                    self.assertTrue(all(value is None for value in metric.group_means.values()))
                    self.assertTrue(all(np.isnan(value).all() for value in metric.layer_means.values()))
                    self.assertTrue(all((value != "not_evaluated").all() for value in metric.layer_reasons.values()))
        provider = mock.Mock(side_effect=AssertionError("invalid task reached provider"))
        for index in (-1, 63, True, 1.0):
            with self.assertRaisesRegex(ValueError, "batch_index"):
                partition.evaluate_draw_batch(provider, TRAIN_IDS, TEST_IDS,
                                               batch_index=index, check=lambda: None)
        with self.assertRaises(ValueError):
            partition.evaluate_draw_batch(provider, TRAIN_IDS, TEST_IDS[:-1], batch_index=0, check=lambda: None)
        provider.assert_not_called()
        failed = mock.Mock(side_effect=MemoryError("caller resource bound"))
        with self.assertRaisesRegex(MemoryError, "resource bound"):
            partition.evaluate_draw_batch(failed, TRAIN_IDS, TEST_IDS, batch_index=62, check=lambda: None)
        failed.assert_called_once()

    def test_incomplete_or_misbound_batches_refuse_and_close_without_retry(self):
        schedule = readonly(np.random.RandomState(42).randint(0, 224, size=(1000, 224)), np.int64)
        make_panel = fixture_factory(schedule)
        point = make_panel(np.arange(224), point=True)
        first = partition.DrawBatch(0, tuple(make_panel(row) for row in schedule[:16]))
        tail = partition.DrawBatch(62, tuple(make_panel(row) for row in schedule[-8:]))
        altered_groups = dict(first.panels[0].pattern.group_means)
        altered_groups["content_minus_brain"] += .125
        bad_summary = replace(first.panels[0], pattern=replace(first.panels[0].pattern,
                                      group_means=MappingProxyType(altered_groups)))
        bad_images = replace(first.panels[0], image_ids=TEST_IDS[::-1])
        bad_layers = replace(first.panels[0], layers=original.LAYERS[::-1])
        for batch_sequence, reason in (
                ((), "all_63"),
                ((first, first), "duplicate"),
                ((replace(tail, panels=tail.panels[:-1]),), "complete_canonical"),
                ((replace(first, batch_index=1),), "draw_order"),
                ((replace(first, panels=first.panels[::-1]),), "draw_order"),
                ((replace(first, panels=(bad_images,) + first.panels[1:]),), "identity"),
                ((replace(first, panels=(bad_layers,) + first.panels[1:]),), "identity"),
                ((replace(first, panels=(bad_summary,) + first.panels[1:]),), "reduction")):
            closed = []

            def completed():
                try:
                    yield from batch_sequence
                finally:
                    closed.append(True)

            with self.assertRaisesRegex(ValueError, reason):
                partition.assemble_bootstrap(point, completed(), check=lambda: None)
            self.assertEqual(closed, [True])

        class Unopened:
            def __init__(self):
                self.opened = self.closed = False

            def __iter__(self):
                self.opened = True
                raise AssertionError("invalid point or denied admission opened batches")

            def close(self):
                self.closed = True

        for supplied, callback, error in (
                (replace(point, image_indices=schedule[0]), lambda: None, ValueError),
                (replace(point, participants=original.PARTICIPANTS[::-1]), lambda: None, ValueError),
                (point, mock.Mock(side_effect=PermissionError("not admitted")), PermissionError)):
            source = Unopened()
            with self.assertRaises(error):
                partition.assemble_bootstrap(supplied, source, check=callback)
            self.assertFalse(source.opened)
            self.assertTrue(source.closed)

        started, closed = [], []

        def cancellation():
            if started:
                raise TimeoutError("resource deadline")

        def interrupted():
            try:
                started.append(True)
                yield first
            finally:
                closed.append(True)
                raise RuntimeError("secondary cleanup failure")

        with self.assertRaisesRegex(TimeoutError, "resource deadline") as failure:
            partition.assemble_bootstrap(point, interrupted(), check=cancellation)
        self.assertEqual(closed, [True])
        self.assertEqual(failure.exception.__notes__, ["batched source cleanup failed: RuntimeError"])


if __name__ == "__main__":
    unittest.main()
