#!/usr/bin/env python3
"""Display the ten complete EP05 comparisons, retaining both animal means."""
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    'primary_reference_summary_attempt001/primary_reference.json',
    'primary_multichannel_summary_attempt001/primary_multichannel.json',
    'primary_capacity_summary_attempt001/primary_capacity.json',
    'primary_temporal_control_summary_attempt001/primary_temporal_control.json',
    'primary_history_control_summary_attempt001/primary_history_control.json',
)
LABELS = ('Spatial-mean LMP', 'Full-channel LMP', 'Low-frequency power',
          'High-frequency power', 'Rank-one LMP', 'RBF LMP',
          'Time-varying LMP', 'Training-pairing control',
          'Four-lag LMP history', 'Channel-label control')


def main():
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator compute allocation required')
    stem = ROOT/'figures/primary_ten_attempt001'
    if any(stem.with_suffix(s).exists() for s in ('.png', '.pdf')):
        raise FileExistsError('Preserve prior figure')
    panels = []
    for source in SOURCES:
        report = json.loads((ROOT/'runs/adaptive_search'/source).read_text())
        if not report['status'].startswith('complete'):
            raise ValueError('Only complete comparisons may be plotted')
        panels.extend(report.get('panels', [report]))
    if len(panels) != 10 or len({p['recipe'] for p in panels}) != 10:
        raise ValueError('All ten distinct registered comparisons required')
    for panel in panels:
        if len(panel['sessions']) != 16:
            raise ValueError('Full sixteen-session panel required')
        for prefix in ('primary', 'identity'):
            means = panel[prefix+'_by_animal']
            if (set(means) != {'Mihili', 'Chewie'} or
                    not np.isclose(np.mean(list(means.values())), panel[prefix+'_increment'], rtol=0, atol=1e-12)):
                raise ValueError('Equal-animal summary differs from reported means')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.3), sharey=True)
    colors = {'Mihili': '#2878a2', 'Chewie': '#c7772b'}
    for ax, prefix, threshold, title in zip(axes, ('primary', 'identity'), (.01, .005),
            ('A  Added full-trajectory prediction', 'B  Correct versus wrong trial pairing')):
        for index, panel in enumerate(panels):
            values = panel[prefix+'_by_animal']
            ax.plot([values['Mihili'], values['Chewie']], [index, index], color='.78', lw=1)
            for animal, offset in (('Mihili', -.09), ('Chewie', .09)):
                ax.scatter(values[animal], index+offset, s=35, color=colors[animal],
                           label=animal if index == 0 else None, zorder=3)
            ax.scatter(panel[prefix+'_increment'], index, s=44, color='#202020', marker='D',
                       label='Equal-animal mean' if index == 0 else None, zorder=4)
        for low, high in ((6.5, 7.5), (8.5, 9.5)):
            ax.axhspan(low, high, color='.96', zorder=0)
        ax.axvline(0, color='.55', lw=.8)
        ax.axvline(threshold, color='#993c59', ls='--', lw=1.3, label=f'Required margin {threshold:g}')
        ax.set_title(title, loc='left', fontsize=12, pad=12)
        ax.set_xlabel('Increment in pooled-trajectory R²')
        ax.grid(axis='x', alpha=.13)
        ax.set_ylim(9.6, -.6)
    axes[0].set_yticks(range(10), LABELS)
    axes[0].set_xlim(-.006, .011)
    axes[1].set_xlim(-.0006, .0073)
    for ax in axes:
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc='upper left', bbox_to_anchor=(.235, .922),
               frameon=False, ncol=3)
    fig.suptitle('EP05 · Trial identity improves before the primary prediction margin is met',
                 x=.235, y=.987, ha='left', fontsize=15)
    fig.text(.235, .945, 'Ten complete comparisons · 16 exposed development sessions · two animals', fontsize=11)
    fig.text(.235, .085, 'Dashed lines: unchanged primary 0.01 and identity 0.005 margins. Gray rows are matched controls.', fontsize=10)
    fig.text(.235, .052, 'Animal means average sessions; black diamonds weight the two animals equally. No uncertainty intervals or future-session claim.', fontsize=9)
    fig.text(.235, .022, 'All ten comparisons are shown. A passing identity margin alone does not qualify a candidate; other required controls remain open.', fontsize=9)
    fig.subplots_adjust(left=.235, right=.985, top=.835, bottom=.17, wspace=.18)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(stem.with_suffix(suffix), dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps({'outputs': [str(stem.with_suffix(s)) for s in ('.png', '.pdf')],
                      'sources': list(SOURCES), 'recipes': [p['recipe'] for p in panels],
                      'new_fits': 0, 'new_source_reads': 0}))


if __name__ == '__main__':
    main()
