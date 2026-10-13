#!/usr/bin/env python3
"""Plot the complete capacity summary and its preserved stationary anchor."""
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Render in coordinator compute')
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'runs/adaptive_search/primary_capacity_summary_attempt001/primary_capacity.json').read_text())
    if report['status'] != 'complete_two_capacity_comparisons':
        raise ValueError('Complete two-capacity comparison required')
    panels = [report['reference_anchor']['panel'], *report['panels']]
    if [p['recipe'] for p in panels] != ['lmp_full_channel_ridge_v1', 'lmp_rank1_ridge_v1', 'lmp_nystrom_rbf105_v1']:
        raise ValueError('Exact three adopted recipes required')
    ids = [r['session_id'] for r in panels[0]['sessions']]
    if len(ids) != 16 or any([r['session_id'] for r in p['sessions']] != ids for p in panels):
        raise ValueError('Same complete sixteen-session support required')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    x = np.arange(3)
    specs = [('delta_R2', 'primary_increment', 'primary_by_animal', .01, 'Improvement over direction/time baseline'),
             ('identity_increment', 'identity_increment', 'identity_by_animal', .005, 'Correct-trial versus wrong-trial pairing')]
    for ax, (row_key, mean_key, animal_key, margin, title) in zip(axes, specs):
        for animal, color in [('Mihili', '#276f94'), ('Chewie', '#bb6632')]:
            for index, row in enumerate(panels[0]['sessions']):
                if row['animal'] == animal:
                    ax.plot(x, [p['sessions'][index][row_key] for p in panels], color=color,
                            alpha=.22, lw=.8, marker='o', ms=3)
            ax.plot(x, [p[animal_key][animal] for p in panels], color=color, lw=2.5,
                    marker='o', ms=6, label=animal+' session mean')
        ax.plot(x, [p[mean_key] for p in panels], color='#202020', lw=1.4,
                ls=':', marker='D', ms=6, label='Equal-animal mean')
        ax.axhline(0, color='.45', lw=.8)
        ax.axhline(margin, color='#754c8c', ls='--', lw=1, label=f'Operating margin {margin:g}')
        ax.set_xticks(x, ['Stationary\nfull LMP ridge', 'Rank-one\nLMP ridge', '105-landmark\nLMP RBF'])
        ax.set_ylabel('Increment in joint X/Y trajectory R²')
        ax.set_title(title, loc='left', fontsize=11.5)
        ax.grid(axis='y', alpha=.14)
        ax.legend(frameon=False, fontsize=8.5, loc='best')
    fig.suptitle('EP05 · Does changing decoder capacity clear the required gains?', x=.07, y=.975, ha='left', fontsize=16)
    fig.text(.07, .92, 'Exposed development only · 16 sessions, 2 animals · unchanged channels, trials, endpoint and outer-session penalty selection', fontsize=10)
    fig.text(.07, .10, 'Faint lines: sessions. Solid lines: animal means. Diamonds: equal-animal mean. No confidence intervals.', fontsize=9.5)
    fig.text(.07, .065, 'All three miss both aggregate operating margins. Kernel and linear penalties are not a matched effective-capacity comparison.', fontsize=9.5)
    fig.text(.07, .03, 'Source: completed primary_capacity_summary_attempt001, including its preserved full-LMP reference. No new fits or source reads.', fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.835, bottom=.23, wspace=.26)
    target = root/'figures/primary_capacity_attempt001'
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(target.with_suffix('.pdf'), bbox_inches='tight')
    print(target.with_suffix('.png'))


if __name__ == '__main__':
    main()
