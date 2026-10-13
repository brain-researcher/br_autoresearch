#!/usr/bin/env python3
"""Render the complete fixed M1 reproduction from its summary, without refits."""
import argparse
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-stem', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use the coordinator compute allocation')
    report = json.loads(args.report.read_text())
    if report['status'] not in ('complete_scoped_benchmark_reconstruction', 'published_support_count_discrepancy'):
        raise ValueError('All sixteen reference sessions must be reconstructed')
    if len(report['sessions']) != 16:
        raise ValueError('Do not plot a successful subset')
    paths = [args.output_stem.with_suffix('.'+s) for s in ('png', 'pdf')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve existing figures')
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    bands = report['upper_band_hz']
    if len(bands) != 9:
        raise ValueError('Expected nine guide-defined LFP bands')
    labels = ['LMP'] + [f'{b:g}' for b in bands[1:]]
    colors = {'Mihili': '#285a90', 'Chewie': '#b45b2b'}
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for session in report['sessions']:
        color = colors[session['animal']]
        style = '--' if session['implant'] == 'Chewie-R' else '-'
        cca = session['channel_mean_CCA']
        decoder = session['mean_XY_split_R_squared']
        axes[0,0].plot(range(9), cca, color=color, alpha=.25, linestyle=style, linewidth=.8)
        axes[0,1].plot(range(10), decoder, color=color, alpha=.25, linestyle=style, linewidth=.8)
        axes[0,2].scatter(cca, decoder[:9], color=color, alpha=.5, s=17, linewidths=0)
    for animal, color in colors.items():
        group = report['group_profiles']['animal:'+animal]
        axes[0,0].plot(range(9), group['session_mean_CCA'], color=color, linewidth=2, label=animal)
        axes[0,1].plot(range(10), group['session_mean_decoder_R_squared'], color=color, linewidth=2)
    axes[0,0].set(title='A  Population association: all sessions', ylabel='Channel-mean CCA')
    axes[0,1].set(title='B  Velocity decoding: all sessions', ylabel='Mean split X/Y correlation squared')
    axes[0,0].legend(frameon=False)
    association = report['association']
    all_x = np.asarray([s['channel_mean_CCA'] for s in report['sessions']]).ravel()
    line_x = np.linspace(all_x.min(), all_x.max(), 100)
    axes[0,2].plot(line_x, association['intercept']+association['slope']*line_x, color='#454545', linewidth=1.3)
    axes[0,2].set(title='C  Published association reconstruction', xlabel='Session/band mean CCA', ylabel='Session/band mean decoding score')
    axes[0,2].text(.03,.96,f"Observed R² = {association['r_squared']:.4f}\nPublished R² = 0.78",transform=axes[0,2].transAxes,va='top')
    for ax, medians, means, title, ylabel in (
        (axes[1,0],report['CCA_caption_medians'],report['CCA_code_means'],'D  Pooled CCA summaries','CCA'),
        (axes[1,1],report['decoder_caption_medians'],report['decoder_code_means'],'E  Pooled decoding summaries','Correlation squared')):
        ax.plot(range(len(medians)),medians,'o-',color='#6e4382',markersize=4,label='Caption: median')
        ax.plot(range(len(means)),means,'s--',color='#28766e',markersize=4,label='Code: mean')
        ax.set(title=title,ylabel=ylabel)
        ax.legend(frameon=False,fontsize=8)
    for row in (0,1):
        axes[row,0].set_xticks(range(9),labels)
        axes[row,1].set_xticks(range(10),labels+['Latent'])
        axes[row,0].set_xlabel('LFP band upper edge (Hz)')
        axes[row,1].set_xlabel('LFP band upper edge (Hz), or latent comparator')
    support = report['support']
    axes[1,2].axis('off')
    axes[1,2].text(0,1,'F  Evidence and limits',fontsize=11,va='top')
    details = (f"16 sessions from two animals\n"
               f"CCA values per band: {support['CCA_values_per_band']} (paper: 606)\n"
               f"Decoding values per representation: {support['decoder_values_per_representation']} (paper: 1,600)\n"
               f"Association points: {support['association_points']} (paper: 144)\n\n"
               "Thin curves: sessions; thick curves: animal means.\n"
               "Dashed thin curves: Chewie's second implant.\n"
               "Both implants belong to one animal.\n\n"
               "Published descriptive statistics; no biological CI.\n"
               "Original split draws are unavailable.\n"
               "Benchmark reconstruction does not authorize search.")
    axes[1,2].text(0,.9,details,va='top',fontsize=9,linespacing=1.5)
    for ax in axes.flat:
        if ax.axison:
            ax.grid(alpha=.13)
            ax.title.set_fontsize(10)
    fig.suptitle('EP05 · Fixed M1 publication benchmark',x=.065,ha='left',fontsize=17)
    fig.text(.065,.94,'All eligible sessions retained · original reference semantics · source: benchmark_comparison.json',fontsize=10)
    fig.subplots_adjust(left=.065,right=.985,top=.87,bottom=.09,wspace=.35,hspace=.45)
    for path in paths:
        fig.savefig(path,dpi=180,facecolor='white')
    plt.close(fig)
    print(json.dumps({'source':str(args.report),'figures':[str(p) for p in paths]}))


if __name__ == '__main__':
    main()
