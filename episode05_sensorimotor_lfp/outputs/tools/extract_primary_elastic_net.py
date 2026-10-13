"""Compact complete-panel elastic-net evidence without refitting or source access."""
import argparse
import json
from pathlib import Path

from extract_primary_relevance import bounds, seed_summaries, thin_panel
from primary_elastic_net import RECIPES


def diagnostics(rows):
    folds = [fold for row in rows for fold in row['selected_fit_diagnostics']]
    output = {'fold_count': len(folds), 'common_scale_range': bounds(
        fold['common_velocity_penalty_scale'] for fold in folds)}
    for arm, fits in (('candidate', [fold['paired'] for fold in folds]),
                      ('controls', [fit for fold in folds for fit in fold['controls']])):
        output[arm] = {key+'_range': bounds(fit[key] for fit in fits) for key in
            ('maximum_KKT_residual', 'KKT_tolerance', 'objective_error_upper_bound',
             'completed_sweeps', 'zero_coefficients', 'active_X', 'active_Y',
             'active_channel_union', 'active_channel_intersection',
             'coefficient_L1_norm', 'coefficient_Frobenius_norm')}
        output[arm]['all_converged'] = all(fit['converged'] for fit in fits)
        output[arm]['all_zero_fit_count'] = sum(fit['active_channel_union'] == 0 for fit in fits)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    if report['status'] != 'complete_elastic_net_control_panel':
        raise ValueError('Complete panel required')
    panels = {panel['recipe']: panel for panel in report['panels']}
    candidate, control = (panels[key] for key in RECIPES)
    if any(len(panel['sessions']) != 16 for panel in (candidate, control)):
        raise ValueError('All sixteen original sessions required')
    keys = ('status', 'scientific_trial_count_increment', 'total_valid_primary_trials_including_prior',
            'post_initial_comparisons', 'post_initial_controls', 'structural_duplicate',
            'regularization_screens', 'all_regularization_screens_pass')
    result = {key: report[key] for key in keys}
    result.update(panels=[thin_panel(candidate), thin_panel(control)],
        candidate_minus_control=thin_panel(report['training_label_control']),
        anchor_contrasts={key: thin_panel(value) for key, value in report['anchor_contrasts'].items()},
        selected_fit_diagnostics=diagnostics(candidate['sessions']),
        control_seeds=seed_summaries(candidate, control))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
