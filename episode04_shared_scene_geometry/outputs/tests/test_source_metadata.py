"""Focused group-integrity and fifteen-cell bookkeeping checks; no real inputs."""
from collections import Counter
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from prepare_source_metadata import CELLS, CUTOFFS, PARTICIPANTS, assign_folds, validate_groups, repeat_metadata, trial_rows


def fixture(repetitions=20):
    images, components = {}, []
    for index in range(repetitions):
        members = []
        for participant, stratum in CELLS:
            key = f'{index:02d}_{participant}_{stratum}'
            images[key] = {'role': 'development_unique', 'participants': [participant],
                           'stratum': stratum, 'planned_repeats': 4}
            members.append(key)
        components.append({'component_id': f'group{index:02d}', 'image_ids': members,
                           'removed_shared_members': 0})
    return images, components


def provider_files(root, groups, pooled=(), by_pool=None, changes=None):
    (root/'data').mkdir()
    payloads = {'pooled_image_groups.json': {'cosine_distance_cutoffs': CUTOFFS, 'groups': groups},
                'pooled_duplicate_groups.json': list(pooled),
                'duplicate_groups.json': by_pool or {}, 'random_fold_changes.json': changes or {}}
    for name, value in payloads.items():
        (root/'data'/name).write_text(json.dumps(value))


class SourceMetadataTests(unittest.TestCase):
    def test_run_local_indices_preserve_global_tsv_axis_and_reject_bad_order(self):
        header = 'session\trun\tbeta_index\tlabel\n'
        valid = 'ses-01\t01\t0\tfirst\nses-01\t01\t1\tsecond\nses-01\t02\t0\tfirst\n'
        rows = list(trial_rows(io.StringIO(header + valid), 'ses-01'))
        self.assertEqual([row['beta_index'] for row in rows], [0, 1, 0])
        self.assertEqual([row['session_beta_axis_index'] for row in rows], [0, 1, 2])
        self.assertEqual([row['label'] for row in rows], ['first', 'second', 'first'])
        for bad in [valid.replace('01\t1', '01\t0'),
                    valid.replace('01\t1', '01\t2'),
                    valid + 'ses-01\t01\t2\tthird\n']:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                list(trial_rows(io.StringIO(header + bad), 'ses-01'))

    def test_missing_repeat_field_keeps_csv_null_separate_from_documented_intent(self):
        blank = repeat_metadata('', 'development_unique')
        self.assertIsNone(blank['planned_repeats'])
        self.assertEqual(blank['n_reps_csv'], '')
        self.assertEqual(blank['intended_repeats'], 4)
        self.assertTrue(blank['intended_repeat_source'].startswith('https://'))
        self.assertEqual(repeat_metadata('12rep', 'excluded_regular_shared')['planned_repeats'], 12)
        with self.assertRaises(ValueError):
            repeat_metadata('', 'conditional_ood')
        with self.assertRaises(ValueError):
            repeat_metadata('3rep', 'development_unique')

    def test_fifteen_cell_greedy_counts_and_group_integrity(self):
        images, components = fixture()
        assignments, group_folds, balance = assign_folds(images, components)
        self.assertEqual(len(assignments), 300)
        self.assertEqual(len(balance), 15)
        for row in balance:
            self.assertEqual(row['fold_counts'], [4]*5)
            self.assertEqual(sum(row['fold_counts']), row['total'])
        for group in components:
            folds = {assignments[key]['fold'] for key in group['image_ids']}
            self.assertEqual(folds, {group_folds[group['component_id']]})
        self.assertEqual([group_folds[f'group{i:02d}'] for i in range(5)], list(range(5)))
        self.assertEqual(assignments, assign_folds(dict(reversed(list(images.items()))), list(reversed(components)))[0])

    def test_actual_mixed_group_bookkeeping(self):
        images, _ = fixture(30)
        flat = sorted(images)
        groups = [{'component_id': 'mixed', 'image_ids': flat[:37], 'removed_shared_members': 2}]
        groups.extend({'component_id': f'single:{key}', 'image_ids': [key],
                       'removed_shared_members': 0} for key in flat[37:])
        assignments, _, balance = assign_folds(images, groups)
        self.assertEqual(len({assignments[key]['fold'] for key in flat[:37]}), 1)
        counted = Counter((images[key]['participants'][0], images[key]['stratum'], row['fold'])
                          for key, row in assignments.items())
        for row in balance:
            self.assertEqual(row['fold_counts'], [counted[(row['participant'], row['source_stratum'], f)] for f in range(5)])

    def test_shared_bridge_retains_global_group(self):
        images, _ = fixture()
        first, second = '00_sub-01_laion', '00_sub-03_laion'
        images['shared_bridge'] = {'role': 'excluded_regular_shared', 'participants': list(PARTICIPANTS),
                                   'stratum': 'laion', 'planned_repeats': 12}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider_files(root, [[first, 'shared_bridge', second]], pooled=[[first, second]])
            groups, _ = validate_groups(images, root)
        retained = next(row for row in groups if row['component_id'] == 'provider_component_00000')
        self.assertEqual(retained['image_ids'], [first, second])
        self.assertEqual(retained['removed_shared_members'], 1)
        self.assertFalse(any('shared_bridge' in row['image_ids'] for row in groups))

    def test_inconsistent_duplicate_or_component_stops(self):
        images, _ = fixture()
        ids = sorted(images)[:4]
        for groups, duplicate in [([ids[:2], ids[1:3]], []),
                                  ([ids[:2], ids[2:]], [[ids[0], ids[2]]])]:
            with self.subTest(groups=groups), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                provider_files(root, groups, pooled=duplicate)
                with self.assertRaises(ValueError):
                    validate_groups(images, root)

    def test_unknown_correction_or_excluded_fold_member_stops(self):
        images, groups = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provider_files(root, [], changes={'sub-01': [{'image_id': 'unknown', 'old_fold': 0, 'new_fold': 1}]})
            with self.assertRaisesRegex(ValueError, 'seeded original'):
                validate_groups(images, root)
        images['excluded'] = {'role': 'conditional_ood', 'participants': list(PARTICIPANTS)}
        groups[0]['image_ids'].append('excluded')
        with self.assertRaisesRegex(ValueError, 'Excluded'):
            assign_folds(images, groups)


if __name__ == '__main__':
    unittest.main()
