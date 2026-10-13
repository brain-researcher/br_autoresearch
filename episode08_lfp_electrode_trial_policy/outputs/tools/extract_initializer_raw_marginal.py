"""Extract a completed raw-initializer comparison without source reads or fits."""
import argparse
import json
from pathlib import Path

from initializer_raw_marginal import POLICY, CONTROL
from initializer_raw_marginal_experiment import PREVIOUS_CONTROL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    report = json.loads(parser.parse_args().report.read_text())
    if (report['status'] != 'completed_two_initializer_raw_marginal_comparisons'
            or len(report['sessions']) != 8 or not report['scientific_scores_released']):
        raise ValueError('Complete released eight-session raw-initializer result required')
    keys = ('status', 'started_utc', 'observed_utc', 'elapsed_seconds', 'process_CPU_seconds',
            'maximum_RSS_KiB', 'fit_progress', 'logical_neural_bytes', 'unique_logical_neural_bytes',
            'fitting_trial_read_calls', 'evaluation_trials_read', 'retained_report_bytes',
            'full_electrode_historical_and_within_run_score_replay_max_absolute_difference',
            'historical_direction_null_Q_replay_max_absolute_difference',
            'prospective_directional_screens', 'valid_evaluations_total', 'held_session_accessed')
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    keys = ('mean_Q', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
            'mean_delta_Q_from_'+CONTROL, 'mean_delta_Q_from_initial_marginal',
            'mean_delta_Q_from_'+PREVIOUS_CONTROL)
    for policy in (POLICY, CONTROL):
        print('PANEL', json.dumps(next(p for p in report['panels'] if p['policy'] == policy)))
    for session in report['sessions']:
        print('SESSION', json.dumps({'filename': session['filename'], 'animal': session['animal'],
            **{key: session['panels'][POLICY][key] for key in keys}}))
    for omission in report['fixed_prediction_session_omission']:
        panel = next(p for p in omission['panels'] if p['policy'] == POLICY)
        print('OMIT', json.dumps({'excluded_session': omission['excluded_session'],
            **{key: panel[key] for key in keys},
            'animals': [{'animal': a['animal'], **{key: a[key] for key in keys}} for a in panel['animals']]}))
    print('SEEDS', json.dumps(report['seed_summaries']))
    records = [r for s in report['sessions'] for r in s['selection_records']]
    true = [r for r in records if r['mc_seed'] is None]
    values = [v for r in true for v in r['marginal_gains_by_physical_channel']]
    print('SELECTION', json.dumps({'all_orders': len(records), 'true_orders': len(true),
        'true_gain_range': [min(values), max(values)], 'true_negative_gain_count': sum(v < 0 for v in values),
        'true_zero_SD_input_columns': sum(len(r['zero_SD_input_columns']) for r in true),
        'true_zero_SD_target_columns': sum(len(r['zero_SD_target_columns']) for r in true)}))


if __name__ == '__main__':
    main()
