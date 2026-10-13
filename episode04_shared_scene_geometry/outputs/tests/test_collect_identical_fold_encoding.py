"""Focused collector checks: exact coverage, negative R2 and equal-weight means."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import collect_identical_fold_encoding as collector
import native_rsa as native


class IdenticalFoldEncodingCollectionTests(unittest.TestCase):
    def test_complete_unit_set_rejects_one_missing_or_duplicate_fold(self):
        reports = [(Path(f'{person}-{fold}.json'), {'participant': person, 'held_fold': fold})
                   for person in collector.panel.PARTICIPANTS for fold in range(5)]
        self.assertEqual(len(collector.unit_map(reports)), 25)
        with self.assertRaises(ValueError):
            collector.unit_map(reports[:-1])
        with self.assertRaises(ValueError):
            collector.unit_map(reports[:-1]+[reports[0]])

    def test_negative_score_is_retained_and_undefined_baseline_is_not_zero(self):
        row = {'complete': True, 'status': 'evaluable', 'processed_voxels': 3, 'retained_voxels': 3,
               'SSE_partial': 8., 'SSE_baseline_partial': 2., 'SSE': 8., 'SSE_baseline': 2., 'R2_SSE': -3.}
        self.assertEqual(collector.checked_score(row), -3.)
        with self.assertRaises(ValueError):
            collector.checked_score(dict(row, R2_SSE=0.))
        with self.assertRaises(ValueError):
            collector.checked_score(dict(row, processed_voxels=2))
        undefined = dict(row, status='not_evaluable', R2_SSE=None,
                         SSE_baseline=0., SSE_baseline_partial=0., reason='Zero baseline')
        self.assertIsNone(collector.checked_score(undefined))

    def test_equal_fold_sector_participant_aggregation_and_explicit_missing_evc(self):
        rows = []
        for p, person in enumerate(collector.panel.PARTICIPANTS):
            for fold in range(5):
                for r, roi in enumerate(collector.panel.ROIS):
                    baseline = -1.+.1*p+.02*fold+.05*r
                    gains = {'DINOv2': 0., 'CLIP': -.2+.01*p+.002*fold+.003*r,
                             'PEcore': .05, 'SigLIP2': -.1}
                    for model in collector.panel.MODELS:
                        rows.append({'participant': person, 'held_fold': fold, 'roi': roi,
                            'model_label': model, 'status': 'evaluable', 'R2_SSE': baseline+gains[model],
                            # Unequal image/voxel/SSE weights may not change biological weighting.
                            'evaluation_images': 10**(p+1), 'retained_voxels': 10**(r+1),
                            'SSE_baseline': 10.**(p+fold+1)})
        primary, evc, effects, evc_rows = collector.summarize(np, native, rows)
        self.assertTrue(primary and evc)
        np.testing.assert_allclose(effects['gain'], [-.173, .05, -.1], atol=1e-14)
        np.testing.assert_allclose(effects['participant'][0], [-.193, -.183, -.173, -.163, -.153], atol=1e-14)
        self.assertAlmostEqual(effects['mean_DINOv2_R2'], -.71, places=14)
        self.assertTrue(all(item['equal_participant_mean_R2'] < 0 for item in evc_rows))
        missing = next(row for row in rows if row['roi'] == collector.panel.ROIS[3]
                       and row['model_label'] == 'CLIP')
        missing.update(status='not_evaluable', R2_SSE=None, reason='Missing EVC repeats')
        primary, evc, incomplete_effects, evc_rows = collector.summarize(np, native, rows)
        self.assertTrue(primary)
        self.assertFalse(evc)
        self.assertEqual(effects, incomplete_effects)
        clip = next(row for row in evc_rows if row['model_label'] == 'CLIP')
        self.assertIsNone(clip['equal_participant_mean_R2'])
        self.assertEqual(len(clip['unavailable_cells']), 1)
        with self.assertRaises(ValueError):
            collector.summarize(np, native, rows[:-1])
        self.assertEqual(collector.direction_comparison(.02, -.1), 'opposite_directions_readout_dependence')


if __name__ == '__main__':
    unittest.main()
