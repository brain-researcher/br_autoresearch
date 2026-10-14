#!/usr/bin/env python3
"""Report existing opposite-half logistic choice scores; no fitting or BOLD access."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal

import numpy as np

EPISODE = Path(__file__).resolve().parents[2]
SOURCE = EPISODE / 'outputs/runs/exploratory_fmri_specification_sensitivity/prepared_20261013_002/behavioral_covariates.json'
HALVES = (((1, 4), (2, 3)), ((2, 3), (1, 4)))
METRICS = ('intercept_log_loss', 'model_log_loss', 'log_loss_improvement')


def participant_scores(metadata):
    if (metadata.get('method') != 'fixed_per_participant_opposite_half_ridge_logistic'
            or metadata.get('features') != ['intercept', 'gain/10', 'positive_loss/10']
            or metadata.get('penalty_diagonal') != [.01, 1., 1.]
            or metadata.get('loss_denominator') != 'responded_trials_within_participant_then_equal_participant_mean'):
        raise ValueError('Require the existing fixed logistic preparation metadata')
    people = metadata.get('participants', {})
    if len(people) != 108 or Counter(row['task_version'] for row in people.values()) != {'EI': 54, 'ER': 54}:
        raise ValueError('Retain all 108 participants, including 54 EI and 54 ER')
    records = []
    for person, record in sorted(people.items()):
        directions = record['directions']
        if len(directions) != 2 or {row['direction'] for row in directions} != {0, 1}:
            raise ValueError(f'{person}: both original scoring directions are required')
        ordered = sorted(directions, key=lambda row: row['direction'])
        counts, model_nll, baseline_nll = [], [], []
        row = {'participant': person, 'task_version': record['task_version']}
        for index, direction in enumerate(ordered):
            if (tuple(direction['train_runs']) != HALVES[index][0]
                    or tuple(direction['score_runs']) != HALVES[index][1]):
                raise ValueError(f'{person}: opposite run halves changed')
            count = direction['scoring_responded']
            if type(count) is not int or not 1 <= count <= 128:
                raise ValueError(f'{person}: invalid responded-trial denominator')
            values = [float(direction[name]) for name in ('model_nll', 'intercept_nll')]
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f'{person}: missing or invalid saved choice log loss')
            for fit_name in ('fit', 'intercept_fit'):
                fit = direction[fit_name]
                if (fit['training_responded'] != ordered[1-index]['scoring_responded']
                        or not math.isfinite(float(fit['gradient_max']))
                        or not 0 <= float(fit['gradient_max']) <= 1e-6):
                    raise ValueError(f'{person}: saved training support or convergence differs')
            counts.append(count)
            model_nll.append(values[0])
            baseline_nll.append(values[1])
            row.update({f'direction_{index}_scoring_responded': count,
                        f'direction_{index}_model_nll': values[0],
                        f'direction_{index}_intercept_nll': values[1]})
        total = sum(counts)
        model, baseline = sum(model_nll)/total, sum(baseline_nll)/total
        computed = {'model_log_loss': model, 'intercept_log_loss': baseline,
                    'log_loss_improvement': baseline-model}
        if record['scoring_responded'] != total or any(
                not math.isclose(float(record[name]), value, rel_tol=1e-12, abs_tol=1e-12)
                for name, value in computed.items()):
            raise ValueError(f'{person}: participant scores disagree with summed half scores')
        row.update(scoring_responded=total, original_trials=256, misses_excluded=256-total,
                   **computed, improved=baseline > model)
        records.append(row)
    return records


def make_figure(output, records, summaries):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {'status': 'unavailable', 'reason': str(exc)}
    colors = {'all': '#555555', 'EI': '#2768a3', 'ER': '#bd6727'}
    figure, (paired, improvement) = plt.subplots(1, 2, figsize=(10, 4.5))
    for version in ('EI', 'ER'):
        panel = [row for row in records if row['task_version'] == version]
        paired.scatter([row['intercept_log_loss'] for row in panel],
                       [row['model_log_loss'] for row in panel], s=22, alpha=.7,
                       color=colors[version], label=version)
    highest = max(row[name] for row in records for name in ('model_log_loss', 'intercept_log_loss'))*1.05
    paired.plot([0, highest], [0, highest], '--', color='#999999', linewidth=1)
    paired.set(xlim=(0, highest), ylim=(0, highest), xlabel='Intercept reference log loss',
               ylabel='Fixed logistic model log loss', title='Paired participant choice predictions')
    paired.set_aspect('equal', adjustable='box')
    paired.legend(frameon=False)
    for index, stratum in enumerate(('all', 'EI', 'ER')):
        panel = [row for row in records if stratum == 'all' or row['task_version'] == stratum]
        values = [row['log_loss_improvement'] for row in panel]
        improvement.scatter(index+np.linspace(-.12, .12, len(panel)), values,
                            color=colors[stratum], alpha=.35, s=13)
        summary = summaries[stratum]['log_loss_improvement']
        lower, upper = summary['t95_CI']
        improvement.errorbar(index, summary['mean'],
                             yerr=[[summary['mean']-lower], [upper-summary['mean']]],
                             fmt='D', color='black', capsize=5, markersize=5)
    improvement.axhline(0, color='#999999', linestyle='--', linewidth=1)
    improvement.set(xticks=[0, 1, 2], xticklabels=['All (108)', 'EI (54)', 'ER (54)'],
                    ylabel='Intercept − logistic log loss (nats/responded trial)',
                    title='Participant improvement; mean and t 95% CI')
    for axis in (paired, improvement):
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Exploratory within-session prediction · both opposite run halves', fontsize=11)
    figure.tight_layout()
    files = []
    for extension in ('png', 'pdf'):
        path = output/f'exploratory_choice.{extension}'
        figure.savefig(path, dpi=180, bbox_inches='tight')
        files.append(path.name)
    plt.close(figure)
    return {'status': 'complete', 'files': files,
            'paired_plot': 'One point per participant; lower model loss indicates better choice prediction.',
            'improvement_plot': 'Each stratum shows its participant values and their mean with an uncorrected Student t95 interval; all contains the same EI/ER participants.'}


def summarize(output):
    from exploratory_fmri_runner import atomic_json
    from summarize_exploratory_fmri import scalar_summary
    if output.exists():
        raise FileExistsError('Use a fresh report output directory')
    # Consume only the saved metadata; the top-level trial rows are not used.
    with SOURCE.open() as stream:
        metadata = json.load(stream)['metadata']
    records = participant_scores(metadata)
    summaries = {}
    for stratum in ('all', 'EI', 'ER'):
        panel = [row for row in records if stratum == 'all' or row['task_version'] == stratum]
        improved = sum(row['improved'] for row in panel)
        ties = sum(row['log_loss_improvement'] == 0 for row in panel)
        summaries[stratum] = {
            'participants': len(panel), 'responded_trials': sum(row['scoring_responded'] for row in panel),
            'misses_excluded': sum(row['misses_excluded'] for row in panel),
            'improved_participants': improved, 'tied_participants': ties,
            'worse_participants': len(panel)-improved-ties, 'proportion_improved': improved/len(panel),
            **{name: scalar_summary([row[name] for row in panel]) for name in METRICS}}
    output.mkdir(parents=True)
    with (output/'participant_choice_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = {
        'analysis_id': 'exploratory_choice_score_report', 'status': 'complete',
        'source': str(SOURCE.resolve()), 'source_component': 'metadata only',
        'participants': 108, 'task_versions': {'EI': 54, 'ER': 54}, 'original_trials': 27648,
        'fitting_performed': False, 'BOLD_access': False, 'directions_per_participant': 2,
        'method': metadata['method'], 'features': metadata['features'],
        'penalty_diagonal': metadata['penalty_diagonal'],
        'half_pairs': [{'train': list(train), 'score': list(score)} for train, score in HALVES],
        'denominator': 'Sum both held-out half NLLs and divide by that participant\'s total responded trials; then give participants equal weight.',
        'positive_improvement': 'Intercept-reference log loss minus logistic-model log loss; positive means the logistic model has lower loss.',
        'summaries': summaries, 'participant_scores': 'participant_choice_scores.csv',
        'interpretation': [
            'Exploratory within-session choice prediction using already completed fixed opposite-half fits; exposed behavior is not fresh confirmation.',
            'Scores concern acceptance conditional on a response. Misses and RT are not modeled or scored; these nats per responded trial must not be compared numerically with the earlier joint choice/RT/deadline score.',
            'Participants are the sampling units. Intervals are uncorrected Student t95 intervals for equal-participant means; improvement uncertainty preserves the within-participant pairing.',
            'Counts with lower loss are descriptive, not psychological-mechanism identification, an established superiority gate, or a novelty claim.',
            'EI/ER are descriptive between-participant strata with different offer ranges, not independent replications or causal task-version effects.'],
        'created_utc': datetime.now(timezone.utc).isoformat()}
    report['figure'] = make_figure(output, records, summaries)
    atomic_json(output/'exploratory_choice_summary.json', report)
    lines = ['# Exploratory opposite-half choice prediction', '',
             'The fixed ridge-logistic model and its regularized intercept reference were already fitted on opposite run halves. This report uses their saved scores for all **108 participants** (54 EI, 54 ER); it performs no new fit.', '',
             'Loss is in **nats per responded trial**. Each participant combines both half-score sums using the actual total responded-trial denominator; group means weight participants equally. Lower loss is better.', '',
             '| Stratum | Intercept mean | Logistic mean | Paired improvement, mean [t95 CI] | Participant SE of improvement | Participants improved |',
             '|---|---:|---:|---|---:|---:|']
    for stratum, panel in summaries.items():
        improvement = panel['log_loss_improvement']
        lower, upper = improvement['t95_CI']
        lines.append(f"| {stratum} | {panel['intercept_log_loss']['mean']:.6f} | {panel['model_log_loss']['mean']:.6f} | {improvement['mean']:+.6f} [{lower:+.6f}, {upper:+.6f}] | {improvement['participant_SE']:.6f} | {panel['improved_participants']}/{panel['participants']} ({panel['proportion_improved']:.1%}) |")
    all_scores = summaries['all']
    lines += ['', f"Scored **{all_scores['responded_trials']:,} responded trials**; **{all_scores['misses_excluded']} misses** do not enter the binary likelihood. Intervals are descriptive and uncorrected for multiple comparisons. EI/ER differences are between-participant descriptions.", '',
              'This choice-only score excludes reaction-time and deadline likelihoods and is not numerically comparable with the earlier joint behavioral scores. Predictive improvement does not identify a valuation or decision-bias mechanism and does not establish novelty or fresh confirmation.', '',
              '[Full summary, including mean/SE/t95 intervals for both model losses](exploratory_choice_summary.json) · [All participant scores and half denominators](participant_choice_scores.csv)', '']
    if report['figure']['status'] == 'complete':
        lines += ['![Saved opposite-half choice predictions](exploratory_choice.png)', '']
    (output/'exploratory_choice_summary.md').write_text('\n'.join(lines))
    return report


def main():
    from exploratory_fmri_runner import require_compute
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=120)
    args = parser.parse_args()
    require_compute()
    if args.max_seconds <= 0:
        raise ValueError('Positive runtime limit required')
    def interrupted(signum, frame):
        raise TimeoutError(f'Saved-score report interrupted by signal {signum}')
    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(args.max_seconds)
    try:
        report = summarize(args.output_dir)
        print(json.dumps({'status': report['status'], 'participants': report['participants'],
                          'summaries': report['summaries']}), flush=True)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
