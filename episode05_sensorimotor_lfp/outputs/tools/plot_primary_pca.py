#!/usr/bin/env python3
"""Plot the completed PCA comparison and preserved matched-lambda LMP anchor."""
import csv
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
        raise RuntimeError('Render in the coordinated compute allocation')
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root/'runs/adaptive_search/primary_pca_control_summary_attempt001/primary_pca.json').read_text())
    if report['status'] != 'complete_pca_control_panel':
        raise ValueError('Complete fixed panel required')
    stem = root/'figures/primary_pca_attempt001'
    if any(stem.with_suffix('.'+ext).exists() for ext in ('png', 'pdf', 'csv')):
        raise FileExistsError('Preserve prior artifacts')
    groups = [report['matched_lambda_anchor_sessions'], report['panels'][0]['sessions'], report['panels'][1]['sessions']]
    names = ('Full LMP', 'PCA90', 'Permuted loadings\nmean of 8 models')
    animals = ('Mihili', 'Chewie')
    colors = {'Mihili': '#276f94', 'Chewie': '#bb6632'}
    keys = ('delta_R2', 'identity_increment')
    rows = []
    ids = [s['session_id'] for s in groups[0]]
    if len(ids) != 16 or any(set(ids) != {s['session_id'] for s in g} for g in groups):
        raise ValueError('All original sixteen sessions must match')
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))
    for j, (ax, key, title, threshold) in enumerate(zip(axes, keys,
            ('A  Improvement over direction/time baseline', 'B  Correct versus wrong trial pairing'), (.01, .005))):
        for animal in animals:
            selected = [s for s in groups[0] if s['animal'] == animal]
            values = np.array([[next(s[key] for s in group if s['session_id'] == old['session_id']) for group in groups] for old in selected])
            if not np.isfinite(values).all():
                raise ValueError('Finite complete scores required')
            for old, vector in zip(selected, values):
                ax.plot(range(3), vector, color=colors[animal], alpha=.23, lw=.8)
                if j == 0:
                    for index, value in enumerate(vector):
                        source = next(s for s in groups[index] if s['session_id'] == old['session_id'])
                        rows.append({'level': 'session', 'animal': animal, 'session': old['session_id'], 'condition': names[index].replace('\n', ' '), **{k: source[k] for k in keys}})
            means = values.mean(axis=0)
            ax.plot(range(3), means, color=colors[animal], lw=2.5, marker='o', ms=7, label=f'{animal}: mean of {len(selected)} sessions')
            if j == 0:
                for index in range(3):
                    rows.append({'level': 'animal_mean', 'animal': animal, 'session': 'all', 'condition': names[index].replace('\n', ' '), **{k: float(np.mean([s[k] for s in groups[index] if s['animal'] == animal])) for k in keys}})
        ax.axhline(0, color='.5', lw=.8)
        ax.axhline(threshold, color='.3', ls='--', lw=1, label=f'Practical margin: {threshold:g}')
        ax.set(title=title, ylabel='Increment in joint XY R²', xticks=range(3), xticklabels=names, xlim=(-.15, 2.15))
        ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle('PCA preserves useful structure but loses to full-channel LMP', fontsize=16, x=.065, ha='left', y=.98)
    fig.text(.065, .025, 'Exposed development evidence · 2 animals · thin lines: all original sessions · no confidence intervals\n'
             'All selected λ = 1. PCA retains 24–44 components; control trace is 25–57% of PCA trace. Equal width does not equal capacity.', fontsize=10, color='.3')
    fig.tight_layout(rect=(.03, .11, .995, .925))
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(stem.with_suffix('.'+ext), dpi=180)
    plt.close(fig)
    with stem.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'figure': str(stem.with_suffix('.png')), 'rows': len(rows), 'source': 'complete PCA summary; no source values or new fits'}))


if __name__ == '__main__':
    main()
