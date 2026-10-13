"""Focused contributor-identity and uncertainty-boundary checks (synthetic only)."""
import json
from pathlib import Path
import tempfile
import unittest

from qualify_group_map_contributors import map_identity, process_task, summarize_support


class ContributorBoundaryTests(unittest.TestCase):
    def test_identity_and_no_inclusion_inference(self):
        identity = map_identity('sub-01_ses-a_contrast-condition_AvB_stat-effect_statmap.nii.gz')
        self.assertEqual(identity['contrast'], 'condition_AvB')
        self.assertEqual(identity['entities'], {'sub': '01', 'ses': 'a'})
        rows = [{**identity, 'path': 'synthetic-effect', 'target_file_present': True}]
        support = summarize_support(['01', '02'], rows)
        self.assertEqual(support['source_output_supported_subject_count'], 1)
        self.assertEqual(support['configured_without_source_output_support'], ['2'])
        self.assertIsNone(support['actual_group_N'])
        self.assertIsNone(support['actual_group_contributor_ids'])
        self.assertFalse(support['source_output_support_is_actual_inclusion'])

    def test_missing_and_multiple_files_do_not_make_more_people(self):
        def row(subject, path, present):
            return {'entities': {'sub': subject}, 'path': path, 'target_file_present': present}
        support = summarize_support(['01', '02'], [row('01', 'a', True), row('1', 'b', True),
                                                   row('02', 'missing', False), row('03', 'c', True)])
        self.assertEqual(support['source_output_supported_subject_count'], 2)
        self.assertEqual(support['subject_ids_with_multiple_matching_effect_paths'], ['1'])
        self.assertEqual(support['unavailable_annex_or_file_paths'], ['missing'])
        self.assertEqual(support['source_output_subjects_outside_configured_input'], ['3'])
        self.assertIsNone(support['actual_group_N'])

    def test_session_grouping_and_positive_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root/'dsTEST/task-demo'
            task_root.mkdir(parents=True)
            spec_path = root/'dsTEST/code/openneuro_glmfitlins/statsmodel_specs/dsTEST/dsTEST-demo_specs.json'
            spec_path.parent.mkdir(parents=True)
            contrast = {'Name': 'AvB', 'ConditionList': ['A', 'B'], 'Weights': [1, -1], 'Test': 't'}
            spec = {'Input': {'subject': ['01', '02']}, 'Nodes': [
                {'Name': 'run_level', 'Level': 'Run', 'Contrasts': [contrast]},
                {'Name': 'subject_level', 'Level': 'Subject', 'GroupBy': ['subject', 'contrast', 'session'],
                 'Model': {'X': [1], 'Type': 'glm'}, 'DummyContrasts': {'Test': 't'}},
                {'Name': 'data_level', 'Level': 'Dataset', 'GroupBy': ['contrast', 'session'],
                 'Model': {'X': [1], 'Type': 'glm'}, 'DummyContrasts': {'Test': 't'}}]}
            spec_path.write_text(json.dumps(spec))
            (task_root/'dataset_description.json').write_text(json.dumps({'PipelineDescription': {
                'Name': 'FitLins', 'Version': 'synthetic', 'Parameters': {'estimator': 'nilearn', 'drop_missing': True}}}))
            for subject, session in [('01', 'a'), ('02', 'b')]:
                path = task_root/f'node-subjectLevel/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_contrast-AvB_stat-effect_statmap.nii.gz'
                path.parent.mkdir(parents=True)
                # These are deliberately not valid images: inspection must only stat.
                path.write_text('not an image; never decode')
            task = {'dataset_id': 'dsTEST', 'task': 'demo', 'task_directory': str(task_root),
                    'existing_spec_paths': [str(spec_path)],
                    'contrast_definition_comparison': [{'contrast': 'AvB', 'receipt_conditions': ['A', 'B'],
                        'receipt_weights': [1, -1], 'exact_conditions_and_weights_match': True}],
                    'map_filename_candidates': [{'contrast': 'AvB', 'declared_node_name': 'data_level',
                        'path': str(task_root/'node-dataLevel/ses-a/ses-a_contrast-AvB_stat-z_statmap.nii.gz')}]}
            first = process_task(task, root)['maps'][0]
            self.assertEqual(first['source_output_supported_subject_ids'], ['1'])
            self.assertEqual(first['provider_declared_statistic_status'], 'provider_declared_signed_group_z')
            self.assertIsNone(first['actual_group_N'])
            task['map_filename_candidates'][0]['path'] = str(task_root/'node-dataLevel/contrast-AvB_stat-z_statmap.nii.gz')
            missing_session = process_task(task, root)['maps'][0]
            self.assertFalse(missing_session['map_session_identity_matches_grouping'])
            self.assertEqual(missing_session['source_output_supported_subject_count'], 0)
            self.assertEqual(missing_session['provider_declared_statistic_status'],
                             'source_statistic_or_sign_route_unresolved')
            spec['Nodes'][-1]['GroupBy'] = ['contrast']
            spec_path.write_text(json.dumps(spec))
            task['map_filename_candidates'][0]['path'] = str(task_root/'node-dataLevel/contrast-AvB_stat-z_statmap.nii.gz')
            pooled = process_task(task, root)['maps'][0]
            self.assertEqual(pooled['source_output_supported_subject_ids'], ['1', '2'])
            spec['Nodes'][-1]['GroupBy'] = []
            spec_path.write_text(json.dumps(spec))
            self.assertEqual(process_task(task, root)['maps'][0]['provider_declared_statistic_status'],
                             'source_statistic_or_sign_route_unresolved')
            spec['Nodes'][-1]['GroupBy'] = ['contrast']
            spec['Nodes'][1]['GroupBy'] = ['subject']
            spec_path.write_text(json.dumps(spec))
            self.assertEqual(process_task(task, root)['maps'][0]['provider_declared_statistic_status'],
                             'source_statistic_or_sign_route_unresolved')
            spec['Nodes'][1]['GroupBy'] = ['subject', 'contrast', 'session']
            spec['Nodes'][1]['Model']['X'] = ['group']
            spec_path.write_text(json.dumps(spec))
            self.assertEqual(process_task(task, root)['maps'][0]['provider_declared_statistic_status'],
                             'source_statistic_or_sign_route_unresolved')


if __name__ == '__main__':
    unittest.main()
