"""Print the completed conditional-initializer contrasts without reopening source."""
import json
from pathlib import Path

from initializer_conditional_gain import POLICY, CONTROL


def main():
    report = json.loads(Path('outputs/runs/development_search/initializer_conditional_attempt001/initializer_conditional_gain.json').read_text())
    if report['status'] != 'completed_two_initializer_conditional_comparisons':
        raise ValueError('Complete eight-session comparison required')
    keys = ('status', 'started_utc', 'observed_utc', 'elapsed_seconds', 'process_CPU_seconds',
            'maximum_RSS_KiB', 'combined_resource_cost', 'combined_fit_progress',
            'retained_report_bytes', 'inherited_pilot_retained_report_bytes',
            'full_electrode_historical_and_within_run_score_replay_max_absolute_difference',
            'historical_direction_null_Q_replay_max_absolute_difference',
            'prospective_directional_screens', 'valid_evaluations_total', 'held_session_accessed')
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    contrasts = ('mean_Q', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
                 'mean_delta_Q_from_'+CONTROL, 'mean_delta_Q_from_initial_marginal',
                 'mean_delta_Q_from_initial_uniform_order5_v1',
                 'mean_delta_Q_from_initial_direction_repeatability_v1')
    for panel in report['panels']:
        print('PANEL', json.dumps({'policy': panel['policy'], 'Q': panel['mean_Q'],
            'identity': panel['mean_pairing_increment'], 'D': panel['mean_delta_Q_from_direction_only']}))
    print('CANDIDATE', json.dumps(next(p for p in report['panels'] if p['policy'] == POLICY)))
    for session in report['sessions']:
        print('SESSION', json.dumps({'filename': session['filename'], 'animal': session['animal'],
            **{key: session['panels'][POLICY][key] for key in contrasts}}))
    for row in report['fixed_prediction_session_omission']:
        panel = next(p for p in row['panels'] if p['policy'] == POLICY)
        print('OMIT', json.dumps({'excluded_session': row['excluded_session'],
            **{key: panel[key] for key in contrasts}, 'animals': [{'animal': a['animal'],
            **{key: a[key] for key in contrasts}} for a in panel['animals']]}))
    print('SEEDS', json.dumps(report['seed_summaries']))


if __name__ == '__main__':
    main()
