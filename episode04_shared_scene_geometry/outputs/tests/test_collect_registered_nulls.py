"""Focused checks of complete coverage and pooled registered-null collection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import collect_registered_nulls as collector
import native_rsa as native


class RegisteredNullCollectionTests(unittest.TestCase):
    def test_disjoint_prefixes_cover_all_cells_and_overlap_or_one_hole_is_rejected(self):
        permutation = np.zeros((5, 5, 8), dtype=bool)
        geometry = np.zeros((5, 5, 3), dtype=bool)
        for person in range(5):
            for fold in range(5):
                collector.claim_range(permutation, person, fold, (0, 2))
                collector.claim_range(permutation, person, fold, (2, 8))
                collector.claim_range(geometry, person, fold, (0, 1))
                collector.claim_range(geometry, person, fold, (1, 3))
        collector.require_complete_coverage(permutation, geometry)
        with self.assertRaises(ValueError):
            collector.claim_range(permutation, 0, 0, (1, 3))
        geometry[4, 4, 2] = False
        with self.assertRaises(ValueError):
            collector.require_complete_coverage(permutation, geometry)
        with self.assertRaises(ValueError):
            collector.declared_range([0, True], 3)

    def test_declared_complete_range_cannot_hide_missing_duplicate_or_extra_score(self):
        def row(draw):
            r = np.full((4, 3), .2)
            return {'family': 'whole_image_permutation', 'draw_index': draw,
                    'model_order': list(collector.unit.MODEL_ORDER),
                    'ROI_order': list(collector.unit.ROI_ORDER),
                    'spearman_r': r.tolist(), 'fisher_z': np.arctanh(r).tolist()}

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'draw_scores.jsonl'
            plans = [(0, 0, (0, 2), (0, 0), path)]
            for draws in ((0,), (0, 0, 1), (0, 1, 2)):
                path.write_text(''.join(json.dumps(row(draw))+'\n' for draw in draws))
                with self.subTest(draws=draws), self.assertRaises(ValueError):
                    collector.collect_scores(np, native, plans, np.full((2, 4, 5, 5, 3), np.nan),
                                             np.empty((0, 3, 5, 5, 3)), lambda: None)
            path.write_text(''.join(json.dumps(row(draw))+'\n' for draw in (1, 0)))
            values = np.full((2, 4, 5, 5, 3), np.nan)
            collector.collect_scores(np, native, plans, values, np.empty((0, 3, 5, 5, 3)), lambda: None)
            np.testing.assert_allclose(values[:, :, 0, 0, :], np.arctanh(.2))

    def test_pooled_argmax_precedes_selection_and_frozen_tie_rule_is_used(self):
        vlm = np.empty((2, 3, 5, 5, 3))
        dino = np.full((5, 5, 3), .1)
        # A large participant-local winner must not replace pooled selection.
        vlm[0, 0] = .2
        vlm[0, 1] = .3
        vlm[0, 2] = -.2
        vlm[0, 2, 0] = 2.
        # Difference within 1e-6 is a tie; lower dimension then lexical ID wins.
        vlm[1, 0] = .4
        vlm[1, 1] = .4000005
        vlm[1, 2] = .4000007
        identities = [native.ModelIdentity('z', 3), native.ModelIdentity('a', 3),
                      native.ModelIdentity('b', 4)]
        selected, statistic, _ = collector.familywise_distribution(
            np, native, vlm, dino, identities, lambda: None)
        np.testing.assert_array_equal(selected, [1, 1])
        np.testing.assert_allclose(statistic, [.3, .4000005], rtol=0, atol=1e-14)
        self.assertNotAlmostEqual(statistic[0], float(vlm[0].mean(axis=(2, 3)).max(axis=0).mean()))

    def test_higher_quantile_keeps_negative_statistics_and_requires_finite_complete_panels(self):
        alignment = np.arange(20, dtype=float)/100-.2
        vlm = np.broadcast_to(alignment[:, None, None, None, None], (20, 3, 5, 5, 3)).copy()
        vlm[:, 1] -= .1
        vlm[:, 2] -= .2
        dino = np.zeros((20, 5, 5, 3))
        identities = [native.ModelIdentity(str(index), 3) for index in range(3)]
        selected, statistic, threshold = collector.familywise_distribution(
            np, native, vlm, dino, identities, lambda: None)
        np.testing.assert_array_equal(selected, np.zeros(20, dtype=np.int8))
        np.testing.assert_allclose(statistic, alignment, rtol=0, atol=1e-14)
        self.assertTrue(np.all(statistic < 0))
        self.assertAlmostEqual(threshold, -.01, places=14)
        self.assertNotAlmostEqual(threshold, float(np.quantile(alignment, .95)), places=4)
        vlm[0, 0, 4, 4, 2] = np.nan
        with self.assertRaises(ValueError):
            collector.familywise_distribution(np, native, vlm, dino, identities, lambda: None)


if __name__ == '__main__':
    unittest.main()
