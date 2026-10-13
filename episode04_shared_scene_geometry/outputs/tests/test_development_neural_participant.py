"""Fixed-grid routing, exact pilot reuse and cumulative resource boundary."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from prepare_development_neural_participant import (
    PARTICIPANTS, ROI_NAMES, require_complete_session, remaining_budget, session_plan,
)


class ParticipantHandoffTest(unittest.TestCase):
    def test_exact_full_grid_reuses_only_the_completed_pilot(self):
        rows = [(person, session, reuse) for person in PARTICIPANTS for session, reuse in session_plan(person)]
        self.assertEqual(len(rows), 150)
        self.assertEqual(len({(person, session) for person, session, _ in rows}), 150)
        self.assertEqual([(p, s) for p, s, reuse in rows if reuse], [('sub-01', 'ses-01')])
        self.assertEqual(sum(not reuse for _, _, reuse in rows), 149)
        with self.assertRaises(ValueError):
            session_plan('sub-02')

    def test_reuse_requires_exact_completed_identity_counts_and_no_protected_decoding(self):
        frozen = {'rows': 1044, 'role_counts': {'development_unique': 575}}
        report = {'complete_neural_session_handoff': True, 'participant': 'sub-01',
            'session': 'ses-01', 'source_root': '/allowed', 'errors': [],
            'roi_names': list(ROI_NAMES), 'selected_trial_count': 575,
            'selected_volumes_decoded': 575, 'total_trial_count': 1044,
            'source_volumes_consumed': 1044, 'excluded_volumes_discarded_as_opaque_bytes': 469,
            'excluded_beta_volumes_numerically_decoded': 0,
            'session_standardization_reliability_RDM_or_model_fit_performed': False,
            'raw_images_or_noise_ceiling_maps_read': False, 'union_voxel_count': 12833,
            'selected_beta_payload_bytes': 29515900, 'planned_payload_upper_bytes': 30718472}
        self.assertEqual(require_complete_session(report, 'sub-01', 'ses-01', Path('/allowed'), frozen), 30718472)
        for key, replacement in (
            ('complete_neural_session_handoff', False), ('participant', 'sub-03'),
            ('session', 'ses-02'), ('selected_volumes_decoded', 574),
            ('source_volumes_consumed', 1043), ('excluded_beta_volumes_numerically_decoded', 1),
            ('selected_beta_payload_bytes', 0), ('errors', ['partial'])):
            changed = copy.deepcopy(report)
            changed[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                require_complete_session(changed, 'sub-01', 'ses-01', Path('/allowed'), frozen)

    def test_cumulative_cap_includes_reused_output_and_stops_at_exhaustion(self):
        self.assertEqual(remaining_budget([30, 40], limit=100), 30)
        with self.assertRaises(ValueError):
            remaining_budget([30, 40, 30], limit=100)
        with self.assertRaises(ValueError):
            remaining_budget([30, 40, 31], limit=100)
        with self.assertRaises(ValueError):
            remaining_budget([30, -1], limit=100)


if __name__ == '__main__':
    unittest.main()
