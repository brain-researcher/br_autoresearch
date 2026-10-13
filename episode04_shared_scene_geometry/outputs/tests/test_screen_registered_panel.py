"""Focused screen boundaries and exclusive publication; no episode payloads."""
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import native_rsa as native
import screen_registered_panel as screen


class RegisteredPanelScreenTests(unittest.TestCase):
    def panels(self):
        values = np.zeros((4, 5, 5, 3))
        for model, gain in (('CLIP', .05), ('PEcore', .04), ('SigLIP2', .03)):
            values[screen.unit.MODEL_ORDER.index(model)] = .1 + gain
        values[screen.unit.MODEL_ORDER.index('DINOv2')] = .1
        partial = np.zeros_like(values)
        for model in screen.panel.VLM_LABELS:
            partial[screen.unit.MODEL_ORDER.index(model)] = .03
        sensitivities = {name: partial.copy() for name in screen.sens.SENSITIVITIES}
        return values, partial, sensitivities

    def decide(self, values, partial, sensitivities, thresholds=(.05, .05)):
        return screen.screen_inputs(np, native, values, partial, sensitivities, thresholds,
            {'identical_fold_encoding': True, 'evc_reliability_context': True})

    def test_partial_loo_failure_rejects_highest_primary_gain(self):
        values, partial, sensitivities = self.panels()
        index = screen.unit.MODEL_ORDER.index('CLIP')
        partial[index, :, :, :] = np.array([-.01, -.01, -.01, -.01, .2])[:, None, None]
        result = self.decide(values, partial, sensitivities)
        self.assertGreater(result['sensitivity_effects']['low_level_partial_rsa']['gain'][0], 0)
        self.assertEqual(result['falsifier_statuses']['low_level_partial_rsa'][0], 'fail')
        self.assertEqual(result['selected_index'], 1)  # PEcore; no rescue of the highest failing row.
        self.assertEqual(len(result['all_participant_effects']), 15)
        self.assertEqual(len(result['all_primary_ROI_effects']), 9)
        self.assertFalse(result['configuration_lock_authorized'])

    def test_zero_sensitivity_gain_is_failure_not_missing_or_pass(self):
        values, partial, sensitivities = self.panels()
        sensitivities['neural_cosine'][:] = 0
        result = self.decide(values, partial, sensitivities)
        self.assertEqual(result['falsifier_statuses']['neural_cosine'], ['fail'] * 3)
        self.assertIsNone(result['selected_index'])

    def test_null_boundary_requires_strict_exceedance(self):
        values, partial, sensitivities = self.panels()
        indices = [screen.unit.MODEL_ORDER.index(m) for m in screen.panel.VLM_LABELS]
        dino = screen.unit.MODEL_ORDER.index('DINOv2')
        exact_largest_alignment = float(native.panel_effects(values[indices], values[dino])['unsubtracted_alignment'].max())
        result = self.decide(values, partial, sensitivities, (exact_largest_alignment, -.5))
        self.assertTrue(all(not gates['image_null'] for gates in result['gates']))
        self.assertIsNone(result['selected_index'])

    def test_incomplete_fold_or_sensitivity_panel_is_not_a_decision(self):
        values, partial, sensitivities = self.panels()
        with self.assertRaises(ValueError):
            self.decide(values[:, :, :4], partial[:, :, :4], sensitivities)
        del sensitivities['complete_repeats']
        with self.assertRaises(ValueError):
            self.decide(values, partial, sensitivities)

    def test_missing_diagnostic_cannot_be_replaced_by_a_passing_primary(self):
        values, partial, sensitivities = self.panels()
        with self.assertRaises(native.NotEvaluable):
            screen.screen_inputs(np, native, values, partial, sensitivities, (.05, .05),
                                 {'identical_fold_encoding': True, 'evc_reliability_context': False})

    def test_outside_collector_path_is_rejected_before_read(self):
        with patch.object(screen.panel, 'read_json', side_effect=AssertionError('must not read')):
            with self.assertRaises(ValueError):
                screen.admit_nulls(None, None, Path('/outside/registered_null_collection.json'),
                                  None, None, None, None, None, lambda: None)
            with self.assertRaises(ValueError):
                screen.admit_encoding(None, None, Path('/outside/identical_fold_encoding_collection.json'),
                                     None, None, None, None, None, None, lambda: None)

    def test_admission_failure_publishes_nothing_and_second_decision_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new-parent/decision.json'
            with self.assertRaises(ValueError):
                screen.publish_decision(path, {'complete_admission': False, 'candidate_decisions': 1}, lambda: None)
            self.assertFalse(path.parent.exists())
            payload = {'complete_admission': True, 'candidate_decisions': 1, 'selected_index': None}
            screen.publish_decision(path, payload, lambda: None)
            saved = path.read_bytes()
            with self.assertRaises(FileExistsError):
                screen.publish_decision(path, dict(payload, selected_index=0), lambda: None)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
