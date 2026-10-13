#!/usr/bin/env python3
"""Plot the completed capacity comparison, without source recordings or refits."""
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
    raise RuntimeError('Use the coordinator compute allocation')
root = Path(__file__).resolve().parents[1]
source = root/'runs/development_search/capacity_comparison_attempt001/capacity_comparison.json'
report = json.loads(source.read_text())
if report['status'] != 'completed_temporal_rank_kernel_development_panels':
    raise ValueError('The complete comparison is required')
destinations = [root/f'figures/development_capacity_attempt001.{s}' for s in ('png', 'pdf')]
if any(p.exists() for p in destinations):
    raise FileExistsError('Preserve existing figures')
methods = ['temporal_ridge', 'reduced_rank', 'rbf']
labels = ['Ten time bins\nLinear ridge', 'Mean features\nReduced-rank ridge', 'Mean features\nRBF kernel ridge']
colors = {'Mihili': '#285a90', 'Chewie': '#b45b2b'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
for ax, field, title, ylabel in zip(axes,
        ['mean_delta_Q_from_mean_residual_ridge', 'mean_pairing_increment'],
        ['Gain over mean-feature residual ridge', 'Dependence on within-direction trial pairing'],
        ['Change in Q', 'Q minus mean wrong-pair Q']):
    for animal, offset in [('Mihili', -.1), ('Chewie', .1)]:
        sessions = [s for s in report['sessions'] if s['animal'] == animal]
        if len(sessions) != 4:
            raise ValueError('Expected four development sessions per animal')
        for index, method in enumerate(methods):
            values = [s['panels'][method][field] for s in sessions]
            ax.scatter(index+offset+np.linspace(-.03,.03,4), values,
                       color=colors[animal], s=28, alpha=.4, linewidths=0)
            ax.scatter(index+offset, np.mean(values), color=colors[animal], marker='D',
                       s=65, edgecolors='white', linewidths=.6,
                       label=f'{animal}: four-session mean' if index == 0 else None)
    ax.axhline(0, color='#555555', linewidth=.8)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-.45,2.45)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc='left')
    ax.grid(axis='y', alpha=.15)
axes[0].legend(frameon=False, loc='upper left', fontsize=9)
fig.suptitle('EP08 · Decoder gains differ between animals', x=.075, ha='left', fontsize=16)
fig.text(.075,.905,'Development evidence · same trial roles and random electrode subsets · four held sessions remain closed',fontsize=10)
fig.text(.075,.08,'Dots: individual sessions; diamonds: animal means. Each value averages the same 12 electrode/trial-count cells.',fontsize=9)
fig.text(.075,.045,'Two animals; no uncertainty intervals. Pairing control preserves direction. Reduced rank has more tuning choices.',fontsize=9)
fig.subplots_adjust(left=.075,right=.985,top=.8,bottom=.25,wspace=.28)
for p in destinations:
    fig.savefig(p,dpi=180,facecolor='white')
plt.close(fig)
print(json.dumps({'source':str(source),'figures':[str(p) for p in destinations]}))
