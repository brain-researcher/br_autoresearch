"""Report the complete high-power comparison and fixed sensitivity contrasts."""
import json
import argparse
from pathlib import Path

from high_power_correspondence import POLICY, CONTROL
from lmp_correspondence import POLICY as LMP_POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    path = parser.parse_args().report
    report = json.loads(path.read_text())
    if report['status'] != 'completed_two_high_power_correspondence_comparisons':
        raise ValueError('Complete eight-session high-power comparison required')
    keys = ('status', 'started_utc', 'observed_utc', 'elapsed_seconds', 'process_CPU_seconds',
            'maximum_RSS_KiB', 'fit_progress', 'retained_report_bytes',
            'logical_neural_bytes', 'unique_logical_neural_bytes', 'fitting_trial_read_calls',
            'evaluation_trials_read', 'within_run_full_high_power_score_replay_max_absolute_difference',
            'historical_direction_null_Q_replay_max_absolute_difference', 'prospective_directional_screens',
            'valid_evaluations_total', 'held_session_accessed', 'correspondence_spectral_contrasts')
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    if report.get('finalization_only'):
        print('FINALIZATION', json.dumps(report['finalization_resources']))
    for panel in report['panels']:
        print('PANEL', json.dumps({'policy': panel['policy'], 'Q': panel['mean_Q'],
            'identity': panel['mean_pairing_increment'], 'D': panel['mean_delta_Q_from_direction_only']}))
    print('CANDIDATE', json.dumps(next(p for p in report['panels'] if p['policy'] == POLICY)))
    contrasts = ('mean_Q', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
                 'mean_delta_Q_from_'+CONTROL, 'mean_delta_Q_from_initial_marginal',
                 'mean_delta_Q_from_'+LMP_POLICY)
    for session in report['sessions']:
        print('SESSION', json.dumps({'filename': session['filename'], 'animal': session['animal'],
            **{key: session['panels'][POLICY][key] for key in contrasts},
            'correspondence_spectral_contrasts': session['correspondence_spectral_contrasts']}))
    for row in report['fixed_prediction_session_omission']:
        panel = next(p for p in row['panels'] if p['policy'] == POLICY)
        print('OMIT', json.dumps({'excluded_session': row['excluded_session'],
            **{key: panel[key] for key in contrasts}, 'animals': [{'animal': a['animal'],
            **{key: a[key] for key in contrasts}} for a in panel['animals']]}))
    print('SEEDS', json.dumps(report['seed_summaries']))


if __name__ == '__main__':
    main()
