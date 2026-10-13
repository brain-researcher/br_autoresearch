"""Summarize the complete logdet comparison without source reads or refits."""
import argparse
import json
from pathlib import Path

from initializer_logdet import POLICY
from initializer_logdet_experiment import UNIFORM, REPEATABILITY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    report = json.loads(parser.parse_args().report.read_text())
    if report['status'] != 'completed_one_initializer_logdet_comparison':
        raise ValueError('Complete eight-session logdet result required')
    keys = ('status', 'started_utc', 'observed_utc', 'elapsed_seconds', 'process_CPU_seconds',
            'maximum_RSS_KiB', 'fit_progress', 'logical_neural_bytes', 'unique_logical_neural_bytes',
            'fitting_trial_read_calls', 'evaluation_trials_read', 'retained_report_bytes',
            'historical_full96_score_replay_max_absolute_difference',
            'historical_direction_null_Q_replay_max_absolute_difference',
            'prospective_directional_screens', 'valid_evaluations_total', 'held_session_accessed')
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    keys = ('mean_Q', 'mean_pairing_increment', 'mean_delta_Q_from_direction_only',
            'mean_delta_Q_from_'+UNIFORM, 'mean_delta_Q_from_'+REPEATABILITY,
            'mean_delta_Q_from_initial_marginal')
    candidate = next(p for p in report['panels'] if p['policy'] == POLICY)
    print('CANDIDATE', json.dumps(candidate))
    for session in report['sessions']:
        print('SESSION', json.dumps({'filename': session['filename'], 'animal': session['animal'],
            **{key: session['panels'][POLICY][key] for key in keys}}))
    for omission in report['fixed_prediction_session_omission']:
        panel = next(p for p in omission['panels'] if p['policy'] == POLICY)
        print('OMIT', json.dumps({'excluded_session': omission['excluded_session'],
            **{key: panel[key] for key in keys}, 'animals': panel['animals']}))
    print('SEEDS', json.dumps(report['uniform_seed_summaries']))
    records = [r for s in report['sessions'] for r in s['selection_records']]
    gains = [g for r in records for g in r['selected_gains']]
    print('SELECTION', json.dumps({'views': len(records), 'gain_range': [min(gains), max(gains)],
        'zero_SD_input_columns': sum(len(r['zero_SD_input_columns']) for r in records),
        'maximum_exact_best_tie_count': max(t for r in records for t in r['available_tie_counts']),
        'numerical_negative_gain_occurrences': sum(r['numerical_negative_gain_occurrences'] for r in records)}))


if __name__ == '__main__':
    main()
