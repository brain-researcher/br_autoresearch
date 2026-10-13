#!/usr/bin/env python3
"""Render completed selector summaries without opening source recordings."""
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
source = root/'runs/development_search/selector_comparison_attempt001/development_selectors.json'
report = json.loads(source.read_text())
if report['status'] != 'completed_five_development_selector_panels' or report['valid_evaluations_charged'] != 5:
    raise ValueError('The complete selected comparison is required')
names = ['direction_repeatability', 'marginal_target', 'conditional_target',
         'permuted_conditional_target', 'diversity']
labels = ['Direction\nrepeatability', 'Marginal target\nassociation', 'Conditional target\nassociation',
          'Permuted-target\nconditional control', 'LFP-only\ndiversity']
destinations = [root/f'figures/development_selectors_attempt001.{suffix}' for suffix in ('png', 'pdf')]
if any(p.exists() for p in destinations):
    raise FileExistsError('Preserve existing figures')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(11.7, 5.4))
colors = {'Mihili': '#285a90', 'Chewie': '#b45b2b'}
for animal, offset in [('Mihili', -.11), ('Chewie', .11)]:
    sessions = [s for s in report['sessions'] if s['animal'] == animal]
    if len(sessions) != 4:
        raise ValueError('Four sessions per animal are required')
    for index, name in enumerate(names):
        values = [s['panels'][name]['delta_Q_from_random_baseline'] for s in sessions]
        ax.scatter(index+offset+np.linspace(-.035, .035, 4), values,
                   color=colors[animal], alpha=.4, s=30, linewidths=0)
        ax.scatter(index+offset, np.mean(values), color=colors[animal], marker='D', s=70,
                   edgecolors='white', linewidths=.6,
                   label=f'{animal}: four-session mean' if index == 0 else None)
ax.axhline(0, color='#555555', linewidth=.8)
ax.set_xticks(range(len(names)), labels)
ax.set_ylabel('Change in Q from matched random subsets')
ax.grid(axis='y', alpha=.15)
ax.set_xlim(-.5, 4.5)
ax.legend(loc='upper right', frameon=False)
fig.suptitle('EP08 · Electrode selection improves development prediction', x=.075, ha='left', fontsize=16)
fig.text(.075, .905, 'Paired differences across the same 12 electrode/trial-count cells in each session', fontsize=11)
fig.text(.075, .075, 'Small dots: sessions; diamonds: animal means. Two animals; no uncertainty intervals or held-session inference.', fontsize=9)
fig.text(.075, .035, 'Training-fold selection only · 5 new panels, 6 total · Global target permutation also breaks direction structure', fontsize=9)
fig.subplots_adjust(left=.075, right=.985, top=.86, bottom=.25)
for path in destinations:
    fig.savefig(path, dpi=180, facecolor='white')
plt.close(fig)
print(json.dumps({'source': str(source), 'figures': [str(p) for p in destinations]}))
