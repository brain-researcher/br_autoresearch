#!/usr/bin/env python3
"""Plot every fixed provider-compression curve from its completed report."""
import argparse
import json
from pathlib import Path
import numpy as np
import run_provider_compression as run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run.base.compute_only()
    report = json.loads(args.report.read_text())
    if (report.get('analysis_id') != 'provider_group_compression' or report.get('complete') is not True
            or set(report['results']) != set(run.arm_menu()) or len(report['contrasts']) != 30):
        raise ValueError('Require the complete fixed 31-arm report')
    if args.out.exists():
        raise FileExistsError('Use a fresh figure output directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = report['results']
    fig, charts = plt.subplots(2, 2, figsize=(13, 9))
    colors = {'provider': '#087f8c', 'global': '#303030', 'shuffle14021': '#bc7137',
              'shuffle14022': '#8472a8', 'shuffle14023': '#82975c', 'raw': '#999999'}
    for column, (family, levels) in enumerate(run.LEVELS.items()):
        top, bottom = charts[0, column], charts[1, column]
        dimensions = [run.GROUP_COUNTS[family]*k for k in levels]
        positions = np.arange(len(levels))
        for variant in ('provider', 'global', *run.VARIANTS[1:]):
            arms = [f'pca{d}' if variant == 'global' else run.arm_name(family, variant, k)
                    for k, d in zip(levels, dimensions)]
            top.plot(positions, [results[a]['external_age']['domain_macro'] for a in arms],
                     'o-', color=colors[variant], label=variant, linewidth=1.5)
        top.axhline(results['raw']['external_age']['domain_macro'], linestyle='--', color=colors['raw'], label='raw ridge')
        top.set_title('Bilateral ROI blocks' if family == 'anatomical' else 'Measurement-family blocks')
        top.set_ylabel('External domain-macro age MAE (years)')
        top.legend(frameon=False, fontsize=8)
        for index, variant in enumerate(('global', *run.VARIANTS[1:], 'raw')):
            rows = []
            for k, dimension in zip(levels, dimensions):
                reference = 'raw' if variant == 'raw' else f'pca{dimension}' if variant == 'global' else run.arm_name(family, variant, k)
                rows.append(next(r for r in report['contrasts'] if r['family'] == family and r['k'] == k and r['reference'] == reference))
            offset = (index-2)*.10
            for x, row in zip(positions+offset, rows):
                low, high = row['descriptive_tailed_95_interval']
                bottom.plot([x, x], [low, high], color=colors[variant])
                bottom.plot(x, row['delta_age'], 'o', color=colors[variant], markersize=4)
        bottom.axhline(0, color='#777777', linewidth=.8)
        bottom.set_ylabel('Provider − control MAE (years)\nNegative favors provider grouping')
        for chart in (top, bottom):
            chart.set_xticks(positions, dimensions)
            chart.set_xlabel('Retained dimensions (fixed; no outer-result selection)')
            chart.spines[['top', 'right']].set_visible(False)
            chart.grid(axis='y', alpha=.2)
    fig.suptitle('EP14 · Provider grouping versus generic compression\nExposed development; all three overlapping RAMP views', fontsize=14)
    fig.text(.04, .025, 'Bottom bars: 95% descriptive intervals from 10,000 paired domain/participant resamples; uncorrected.\n'
             'No whitening or post-PCA rescaling; seven ridge penalties per fixed arm. All shuffles and dimensions retained.\n'
             'See full report for every domain, participant-average, worst-domain, age-quartile and site-probe result. Validation remains closed.', fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .94))
    args.out.mkdir(parents=True)
    for extension in ('png', 'pdf'):
        fig.savefig(args.out/f'provider_compression.{extension}', dpi=170)
    plt.close(fig)
    print(json.dumps({'complete': True, 'source': str(args.report), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
