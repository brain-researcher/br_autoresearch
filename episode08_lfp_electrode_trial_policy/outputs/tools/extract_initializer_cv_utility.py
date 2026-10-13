"""Read the completed CV-utility report; no source reads or model refits."""
import argparse
import json
from pathlib import Path

from initializer_cv_utility import POLICY, CONTROL
from initializer_cv_utility_experiment import UNIFORM, REPEATABILITY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    report = json.loads(parser.parse_args().report.read_text())
    if (report['status'] != 'completed_two_initializer_cv_utility_comparisons'
            or len(report['sessions']) != 8 or not report['scientific_scores_released']):
        raise ValueError('Complete released eight-session result required')
    keys = ('status', 'started_utc', 'observed_utc', 'elapsed_seconds', 'process_CPU_seconds',
            'maximum_RSS_KiB', 'combined_fit_progress', 'combined_resource_cost',
            'retained_report_bytes', 'inherited_pilot_retained_report_bytes',
            'full_electrode_historical_and_within_run_score_replay_max_absolute_difference',
            'historical_direction_null_Q_replay_max_absolute_difference',
            'prospective_directional_screens', 'valid_evaluations_total', 'held_session_accessed')
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    keys = ('mean_Q', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
            'mean_delta_Q_from_'+CONTROL, 'mean_delta_Q_from_'+UNIFORM,
            'mean_delta_Q_from_'+REPEATABILITY, 'mean_delta_Q_from_initial_marginal')
    for policy in (POLICY, CONTROL):
        print('PANEL', json.dumps(next(p for p in report['panels'] if p['policy'] == policy)))
    for session in report['sessions']:
        print('SESSION', json.dumps({'filename': session['filename'], 'animal': session['animal'],
            **{key: session['panels'][POLICY][key] for key in keys}}))
    for omission in report['fixed_prediction_session_omission']:
        panel = next(p for p in omission['panels'] if p['policy'] == POLICY)
        print('OMIT', json.dumps({'excluded_session': omission['excluded_session'],
            **{key: panel[key] for key in keys}, 'animals': panel['animals']}))
    print('SEEDS', json.dumps(report['seed_summaries']))
    records = [r for s in report['sessions'] for r in s['selection_records']]
    true = [r for r in records if r['mc_seed'] is None]
    print('SELECTION', json.dumps({'all_orders': len(records), 'true_orders': len(true),
        'true_utility_range': [min(v for r in true for v in r['signed_validation_utilities']),
                               max(v for r in true for v in r['signed_validation_utilities'])],
        'true_negative_utility_counts_by_view': [sum(v < 0 for v in r['signed_validation_utilities'])
                                                 for r in true],
        'true_negative_selected_counts_by_view': [sum(
            r['signed_validation_utilities'][r['physical_channels'].index(e)] < 0
            for e in r['selected_channel_order32']) for r in true]}))


if __name__ == '__main__':
    main()
