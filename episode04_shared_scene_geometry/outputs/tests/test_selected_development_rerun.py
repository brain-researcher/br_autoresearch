"""Focused adapter checks; run inside the substantive coordinated allocation."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import run_selected_development_rerun as rerun


def decision(model='SigLIP2'):
    index, _, _ = rerun.coordinates(model)
    return {'analysis_id': 'candidate_decision', 'mode': 'complete_registered_fixed_panel_screen',
        'status': 'selected_candidate_pending_exact_full_rerun', 'complete_admission': True,
        'candidate_decisions': 1, 'candidate_selection_performed': True,
        'exact_full_development_rerun_performed': False, 'candidate_lock_authorized': False,
        'configuration_lock_authorized': False, 'OOD_access_authorized': False, 'OOD_outcomes_opened': False,
        'selected_index': index, 'selected_model': {'model_label': model,
            'model_id': rerun.panel.MODELS[model][1], 'dimension': rerun.panel.MODELS[model][0]},
        'VLM_order': list(rerun.panel.VLM_LABELS), 'participant_order': list(rerun.panel.PARTICIPANTS),
        'fold_order': list(range(5)), 'primary_ROI_order': list(rerun.panel.ROIS[:3]),
        'gates': [dict.fromkeys(rerun.GATE_NAMES, True) for _ in range(3)],
        'observed_report': str(rerun.OBSERVED), 'sensitivity_report': str(rerun.SENSITIVITIES),
        'null_report': str(rerun.NULL_REPORT),
        'diagnostics': {'identical_fold_encoding': True, 'evc_reliability_context': True},
        'first_panel_evaluation_utc': '2020-01-01T00:00:00+00:00',
        'finished_utc': '2020-01-01T01:00:00+00:00'}


class SelectedRerunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np

    def test_only_saved_selected_candidate_can_enter(self):
        record = decision()
        self.assertEqual(rerun.admit_decision(record), 'SigLIP2')
        for field, value in (('status', 'closed_no_candidate'), ('complete_admission', False),
                             ('candidate_decisions', 2), ('candidate_decisions', True),
                             ('configuration_lock_authorized', True),
                             ('OOD_access_authorized', True), ('selected_index', 0)):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(record)
                invalid[field] = value
                with self.assertRaises(ValueError):
                    rerun.admit_decision(invalid)
        invalid = copy.deepcopy(record)
        invalid['selected_model']['model_id'] = 'a-different-release'
        with self.assertRaises(ValueError):
            rerun.admit_decision(invalid)

    def test_geometry_seed_index_stays_in_original_three_model_order(self):
        self.assertEqual(rerun.coordinates('CLIP'), (0, 0, 1))
        self.assertEqual(rerun.coordinates('PEcore'), (1, 2, 1))
        self.assertEqual(rerun.coordinates('SigLIP2'), (2, 3, 1))
        # Selected-only arrays must not renumber SigLIP2's seeds to CLIP's.
        vi, _, _ = rerun.coordinates('SigLIP2')
        self.assertEqual(100000 + 1000 * vi + 999, 102999)
        with self.assertRaises(ValueError):
            rerun.coordinates('DINOv2')
        self.assertEqual([rerun.slot_cell(slot) for slot in range(25)],
                         [(person, held) for person in rerun.panel.PARTICIPANTS for held in range(5)])
        for invalid in (-1, 25, True):
            with self.assertRaises(ValueError):
                rerun.slot_cell(invalid)

    def test_null_splice_retains_other_models_and_refreshes_shared_dino(self):
        np = self.np
        with mock.patch.multiple(rerun, PERMUTATIONS=2, GEOMETRIES=1):
            permutations = np.arange(2 * 4 * 5 * 5 * 3, dtype=float).reshape(2, 4, 5, 5, 3) / 10000
            geometries = np.arange(3 * 5 * 5 * 3, dtype=float).reshape(1, 3, 5, 5, 3) / 10000
            before_p, before_g = permutations.copy(), geometries.copy()
            fresh_p = np.stack((permutations[:, 3, 2, 4], permutations[:, 1, 2, 4]), axis=1)
            fresh_g = geometries[:, 2, 2, 4].copy()
            fresh_p += 2e-13
            fresh_g -= 2e-13
            rerun.replace_null_unit(np, permutations, geometries, 'SigLIP2', 2, 4, fresh_p, fresh_g)
            np.testing.assert_array_equal(permutations[:, 3, 2, 4], fresh_p[:, 0])
            np.testing.assert_array_equal(permutations[:, 1, 2, 4], fresh_p[:, 1])
            np.testing.assert_array_equal(geometries[:, 2, 2, 4], fresh_g)
            np.testing.assert_array_equal(permutations[:, [0, 2]], before_p[:, [0, 2]])
            np.testing.assert_array_equal(geometries[:, [0, 1]], before_g[:, [0, 1]])
            # A discrepancy or incomplete draw bank fails before any substitution.
            for broken in (fresh_p + 1e-5, fresh_p[:1], np.full_like(fresh_p, np.nan)):
                current = permutations.copy()
                with self.assertRaises(ValueError):
                    rerun.replace_null_unit(np, permutations, geometries, 'SigLIP2', 2, 4, broken, fresh_g)
                np.testing.assert_array_equal(permutations, current)

    def test_registered_gate_boundaries_without_candidate_reselection(self):
        np = self.np
        effects = {'gain': np.float64(.02), 'participant': np.array([.01, .01, .01, .01, -.001]),
                   'participant_loo': np.full(5, .01), 'sector_loo': np.full(3, .01),
                   'unsubtracted_alignment': np.float64(.2)}
        partial = {'gain': np.float64(.001), 'participant_loo': np.full(5, .001)}
        sensitivities = {name: {'gain': np.float64(.001)} for name in rerun.sens.SENSITIVITIES}
        gates = rerun.selected_gates(np, effects, partial, sensitivities, (.1, .2))
        self.assertTrue(gates['practical_gain'])
        self.assertTrue(gates['participant_direction'])
        self.assertFalse(gates['geometry_null'])  # Exceedance is strict.
        partial['participant_loo'][0] = 0
        effects['participant'][0] = 0
        gates = rerun.selected_gates(np, effects, partial, sensitivities, (.1, .19))
        self.assertFalse(gates['participant_direction'])
        self.assertFalse(gates['low_level_partial_rsa'])
        self.assertEqual(len(gates), 12)

    def test_missing_twenty_fifth_unit_cannot_form_a_familywise_result(self):
        np = self.np
        record = decision('CLIP')
        rois = rerun.panel.ROIS[:3]
        pair = ('CLIP', 'DINOv2')
        deadline = '2020-01-06T00:00:00+00:00'
        report = {'selected_model': record['selected_model'], 'original_panel_deadline_utc': deadline,
                  'unit_reports': [], 'complete_rerun': False}
        observed, observed_rows, sensitivity_rows = {'folds': []}, {}, {}
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(rerun, 'FAMILY', Path(temporary)), \
                mock.patch.multiple(rerun, PERMUTATIONS=2, GEOMETRIES=1):
            units = {}
            for person in rerun.panel.PARTICIPANTS:
                for held in range(5):
                    identity = '/fixture/' + person + '-fold-' + str(held)
                    support = {'participant': person, 'held_fold': held, 'image_id_file': identity,
                               'evaluation_images': 9, 'source_counts': {'laion': 3, 'things': 3, 'thingsplus': 3}}
                    observed['folds'].append(support)
                    primary, sensitivities = [], []
                    for roi in rois:
                        for model in pair:
                            values = dict.fromkeys(('native_spearman_r', 'native_fisher_z',
                                                    'partial_spearman_r', 'partial_fisher_z'), 0.0)
                            primary.append({'roi': roi, 'model_label': model, **values})
                            observed_rows[(person, held, roi, model)] = values
                    for name in rerun.sens.SENSITIVITIES:
                        for roi in rois:
                            for model in rerun.panel.MODELS:
                                sensitivity_rows[(name, person, held, roi, model)] = {'fisher_z': 0.0}
                                if model in pair:
                                    sensitivities.append({'sensitivity': name, 'roi': roi, 'model_label': model, 'fisher_z': 0.0})
                    units[rerun.FAMILY / 'units' / f'{person}-fold-{held}' / rerun.REPORT_NAME] = {
                        'status': 'selected_development_rerun_unit_complete', 'complete_unit': True, 'errors': [],
                        'participant': person, 'held_fold': held, 'selected_model': record['selected_model'],
                        'decision_report': str(rerun.DECISION), 'original_panel_deadline_utc': deadline,
                        'first_panel_evaluation_utc': record['first_panel_evaluation_utc'],
                        'started_utc': '2020-01-01T02:00:00+00:00', 'finished_utc': '2020-01-01T03:00:00+00:00',
                        'completed_permutation_draws': 2, 'completed_geometry_draws': 1,
                        'component_order': list(pair), 'random_geometry_model_index': 0,
                        'component_files': {'permutation': 'permutation_components.npy', 'geometry': 'geometry_components.npy'},
                        'evaluation_image_id_file': identity, 'evaluation_images': 9, 'source_counts': support['source_counts'],
                        'primary_rows': primary, 'sensitivity_rows': sensitivities}
            missing = rerun.FAMILY / 'units/sub-07-fold-4' / rerun.REPORT_NAME
            def read(path):
                if path == missing:
                    raise FileNotFoundError(str(path))
                return units[path]
            arrays = {'observed_component_z': np.zeros((4, 5, 5, 3)),
                      'observed_partial_component_z': np.zeros((4, 5, 5, 3)),
                      'permutation_component_z': np.zeros((2, 4, 5, 5, 3)),
                      'geometry_component_z': np.zeros((1, 3, 5, 5, 3))}
            def load(path, **_):
                return np.zeros((2, 2, 3)) if path.name == 'permutation_components.npy' else np.zeros((1, 3))
            with mock.patch.object(rerun, 'fixed_json', side_effect=read), \
                    mock.patch.object(np, 'load', side_effect=load), \
                    mock.patch.object(rerun.nullcollector, 'familywise_distribution') as familywise:
                with self.assertRaises(FileNotFoundError):
                    rerun.collect_units(np, mock.Mock(), report, Path(temporary), record,
                                        (observed, observed_rows, sensitivity_rows, arrays), lambda: None, lambda _: None)
                familywise.assert_not_called()
            self.assertFalse(report['complete_rerun'])
            self.assertEqual(len(report['unit_reports']), 24)

    def test_existing_unit_or_collection_is_never_replayed_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(rerun, 'FAMILY', Path(temporary)):
            unit = rerun.claim_output('unit', 'sub-03', 2)
            sentinel = unit / 'retained.txt'
            sentinel.write_text('preserve this completed or partial unit')
            with self.assertRaises(FileExistsError):
                rerun.claim_output('unit', 'sub-03', 2)
            self.assertEqual(sentinel.read_text(), 'preserve this completed or partial unit')
            rerun.claim_output('collect')
            with self.assertRaises(FileExistsError):
                rerun.claim_output('collect')


if __name__ == '__main__':
    unittest.main()
