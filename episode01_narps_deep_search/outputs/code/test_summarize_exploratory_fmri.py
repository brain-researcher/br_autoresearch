"""One bounded algebra/support fixture for the full-cohort collector; Slurm only."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import summarize_exploratory_fmri as summary


class FullCohortSummaryTest(unittest.TestCase):
    def test_paired_contrasts_common_support_and_no_complete_case_rescue(self):
        # Exact vectors make signs and run-half averaging independently checkable.
        shape = np.array([1., 2., 3.])
        levels = np.array([10., 7., 6., 2., 5.])
        effects = levels[:, None, None]*np.ones((1, 2, 1))*shape[None, None, :]
        axes = summary.axis_maps(effects)
        expected = np.array([2., 8., 3.5, 4.5, -1., 3.])
        np.testing.assert_allclose(axes[:, 0, 0], expected)
        np.testing.assert_allclose(axes[2]+axes[3], axes[1])
        halves = np.stack([effects, effects*2])
        maps, metrics = summary.participant_metrics(halves)
        np.testing.assert_allclose(maps[2], axes*1.5)
        cross = [row for row in metrics if row['metric'] == 'uncentered_spatial_cosine']
        np.testing.assert_allclose([row['value'] for row in cross], 1.)
        relative = next(row for row in metrics if row['focal'] == 'gain' and row['panel'] == 'all4'
                        and row['metric'] == 'displacement_RMS_over_symmetric_mean_RMS')
        self.assertAlmostEqual(relative['value'], 8./6.)
        missing = effects.copy()
        missing[1] = np.nan
        incomplete = summary.axis_maps(missing)
        self.assertTrue(np.isfinite(incomplete[[0, 1, 5]]).all())
        self.assertTrue(np.isnan(incomplete[[2, 3, 4]]).all())
        moments = summary.MapMoments(3)
        for values in (shape, shape*2, np.full(3, np.nan)):
            moments.add(values)
        self.assertIsNone(moments.estimates(3))
        mean, se = moments.estimates(2)
        np.testing.assert_allclose(mean, shape*1.5)
        np.testing.assert_allclose(se, shape*.5)
        np.testing.assert_array_equal(summary.common_positions(np.array([2, 4, 7, 10]), np.array([4, 10])), [1, 3])
        with self.assertRaisesRegex(ValueError, 'no support filtering'):
            summary.common_positions(np.array([2, 4, 7, 10]), np.array([4, 8]))

        people = [f'sub-{index:03d}' for index in range(108)]
        versions = {person: ('EI' if index < 54 else 'ER') for index, person in enumerate(people)}
        runs = [{'participant': person, 'run': run, 'task_version': versions[person]}
                for person in people for run in (1, 2, 3, 4)]
        inventory = {'status': 'complete', 'participants': people, 'runs': runs}
        self.assertEqual(len(summary.validate_inventory(inventory)[1]), 432)
        with self.assertRaisesRegex(ValueError, 'all 108 participants and 432 runs'):
            summary.validate_inventory({**inventory, 'runs': runs[:-1]})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root/'prepared'
            prepared.mkdir()
            (prepared/'prepared.json').write_text(json.dumps(inventory))
            output = root/'incomplete_report'
            report = summary.summarize(prepared, [root/'no_completed_runs'], output,
                                       root/'unused_membership.tsv', lambda: None)
            self.assertEqual(report['status'], 'incomplete_cohort')
            self.assertEqual(len(report['missing_runs']), 432)
            self.assertFalse((output/'maps').exists())
            self.assertFalse((output/'participant_metrics.csv').exists())
        memberships = {person: f'{versions[person]}-offers-{index%8+1:02d}'
                       for index, person in enumerate(people)}
        records = [{'participant': person, 'axis': 'total_displacement', 'focal': 'gain',
                    'panel': 'half14_vs_half23', 'metric': 'uncentered_spatial_cosine',
                    'value': float(index)} for index, person in enumerate(people)]
        summarized, omissions = summary.group_metric_summaries(records, versions, memberships)
        overall = next(row for row in summarized if row['stratum'] == 'all')
        self.assertEqual(overall['expected_participants'], 108)
        self.assertAlmostEqual(overall['mean'], 53.5)
        self.assertEqual(len(omissions), 32)  # 16 group omissions, all + matched version.
        for row in omissions:
            retained = [record['value'] for record in records
                        if memberships[record['participant']] != row['omitted_offer_sequence']
                        and (row['stratum'] == 'all' or versions[record['participant']] == row['stratum'])]
            self.assertAlmostEqual(row['mean'], np.mean(retained))
        records[0]['value'] = float('nan')
        summarized, _ = summary.group_metric_summaries(records, versions, memberships)
        for stratum, count in (('all', 107), ('EI', 53)):
            row = next(row for row in summarized if row['stratum'] == stratum)
            self.assertEqual(row['available_participants'], count)
            self.assertIsNone(row['mean'])
        self.assertEqual(next(row for row in summarized if row['stratum'] == 'ER')['status'], 'complete')


if __name__ == '__main__':
    summary.require_compute()
    unittest.main()
