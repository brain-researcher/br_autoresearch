#!/usr/bin/env python3
"""Plot all six fixed fMRI axes from saved participant summary statistics only.

No fitting, resampling, metric calculation, map loading or cohort filtering.
Dependencies: Python standard library and matplotlib (with its NumPy dependency).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket


AXES = (
    ('S11', 'S11: full-model coefficient\nAbsolute map'),
    ('total_displacement', 'Total specification difference\nS00 − S11'),
    ('decision_omission', 'Decision-covariate omission\n½[(S00 − S10) + (S01 − S11)]'),
    ('response_omission', 'Response-covariate omission\n½[(S00 − S01) + (S10 − S11)]'),
    ('interaction', 'Factorial interaction\nS00 − S10 − S01 + S11'),
    ('response_deadline_timing', 'Response/deadline timing difference\nR11 − S11'),
)
FOCAL = ('gain', 'loss')
STRATA = ('all', 'EI', 'ER')
COUNTS = {'all': 108, 'EI': 54, 'ER': 54}
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
PANELS = (
    ('uncentered_spatial_cosine', 'half14_vs_half23', 'cross_half_cosine',
     'Cross-half spatial agreement', 'Mean participant uncentered spatial cosine'),
    ('map_RMS', 'all4', 'four_run_map_rms',
     'Magnitude after averaging all four runs', 'Mean participant coefficient-map RMS'),
)


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Use a verified Slurm compute shell')
    if any(os.environ.get(name) != '1' for name in THREADS):
        raise RuntimeError('All numerical thread limits must equal one')


def saved_rows(report):
    """Select the entire fixed panel and validate, without deriving statistics."""
    if (report.get('analysis_id') != 'exploratory_fmri_summary'
            or report.get('status') != 'complete_cohort'
            or report.get('participants') != 108 or report.get('complete_runs') != 432
            or report.get('task_versions') != {'EI': 54, 'ER': 54}):
        raise ValueError('Require the saved complete 108-participant, 432-run fMRI summary')
    wanted = {(axis, focal, stratum, metric, panel)
              for axis, _ in AXES for focal in FOCAL for stratum in STRATA
              for metric, panel, *_ in PANELS}
    selected = {}
    for row in report['metric_summaries']:
        key = tuple(row[name] for name in ('axis', 'focal', 'stratum', 'metric', 'panel'))
        if key not in wanted:
            continue
        if key in selected:
            raise ValueError(f'Duplicate saved metric row: {key}')
        expected = COUNTS[row['stratum']]
        available = row['available_participants']
        missing = row['unavailable_participants']
        if (row['expected_participants'] != expected or type(available) is not int
                or type(missing) is not int or min(available, missing) < 0
                or available + missing != expected):
            raise ValueError(f'Changed participant denominator: {key}')
        if row['status'] == 'complete':
            mean, se = row['mean'], row['participant_SE']
            interval = row['t95_CI']
            if (available != expected or not isinstance(interval, list) or len(interval) != 2
                    or not all(isinstance(value, (int, float)) and math.isfinite(value)
                               for value in (mean, se, *interval))
                    or se < 0 or not interval[0] <= mean <= interval[1]):
                raise ValueError(f'Invalid saved participant summary: {key}')
        elif (row['status'] != 'unavailable_without_complete_case_filtering'
              or any(row[name] is not None for name in ('mean', 'participant_SE', 't95_CI'))):
            raise ValueError(f'Unrecognized unavailable metric: {key}')
        selected[key] = row
    if set(selected) != wanted:
        raise ValueError(f'Missing fixed-panel summary rows: {sorted(wanted - set(selected))}')
    return selected


def render(summary, output):
    raw = summary.read_bytes()
    report = json.loads(raw)
    rows = saved_rows(report)
    if output.exists():
        raise FileExistsError('Use a fresh figure output directory')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output.mkdir(parents=True)
    colors = {'all': '#333333', 'EI': '#286ca5', 'ER': '#bd6727'}
    markers = {'all': 'o', 'EI': 's', 'ER': '^'}
    offsets = {'all': -.21, 'EI': 0., 'ER': .21}
    files = []
    support = report.get('spatial_support', {})
    mask_note = f"One common support: {report['group_voxels']:,} voxels."
    if support.get('amended_after_observed_input_failures'):
        mask_note += ' Numerically valid support approved after input failures.'
    failed = report.get('design_quality', {}).get('failed_run_arm_count')
    quality_note = (f' Saved design-quality failures: {failed} run/arm combinations.'
                    if failed is not None else '')
    for metric, panel, basename, title, xlabel in PANELS:
        figure, charts = plt.subplots(1, 2, figsize=(15, 8.2), sharey=True)
        for focal, chart in zip(FOCAL, charts):
            chart.axhspan(-.5, .5, color='#edf1f5', zorder=0)
            chart.axhline(.5, color='#bbc3cc', linewidth=.9)
            chart.axvline(0, color='#bbbbbb', linewidth=.8, zorder=0)
            for ai, (axis, _) in enumerate(AXES):
                for stratum in STRATA:
                    row = rows[axis, focal, stratum, metric, panel]
                    position = ai + offsets[stratum]
                    if row['status'] == 'complete':
                        lower, upper = row['t95_CI']
                        chart.errorbar(row['mean'], position,
                                       xerr=[[row['mean'] - lower], [upper - row['mean']]],
                                       fmt=markers[stratum], color=colors[stratum],
                                       markersize=5, capsize=2.5, linewidth=1.25)
                    else:
                        chart.text(.98, position,
                                   f"{stratum}: unavailable ({row['available_participants']}/{COUNTS[stratum]})",
                                   transform=chart.get_yaxis_transform(), ha='right', va='center',
                                   fontsize=8, color=colors[stratum])
            chart.set_yticks(range(len(AXES)), [label for _, label in AXES])
            chart.set_ylim(len(AXES) - .5, -.5)
            chart.set_title('Gain slope' if focal == 'gain' else 'Positive-loss slope', fontsize=12)
            chart.set_xlabel(xlabel, fontsize=10)
            chart.tick_params(axis='y', length=0, labelsize=9)
            chart.spines[['top', 'right', 'left']].set_visible(False)
            chart.grid(axis='x', color='#eeeeee', linewidth=.7)
            if metric == 'uncentered_spatial_cosine':
                complete = [rows[axis, focal, stratum, metric, panel]
                            for axis, _ in AXES for stratum in STRATA
                            if rows[axis, focal, stratum, metric, panel]['status'] == 'complete']
                lower = min([-1.] + [row['t95_CI'][0] for row in complete])
                upper = max([1.] + [row['t95_CI'][1] for row in complete])
                chart.set_xlim(lower - .025, upper + .025)
        handles = [Line2D([], [], color=colors[stratum], marker=markers[stratum],
                          linestyle='-', label=f'{stratum.capitalize() if stratum == "all" else stratum} (n={COUNTS[stratum]})')
                   for stratum in STRATA]
        figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.61, .93),
                      ncol=3, frameon=False)
        figure.suptitle('Exploratory fMRI specification sensitivity · ' + title,
                        fontsize=14, x=.58, y=.98)
        measure_note = ('Halves: runs 1+4 versus 2+3; cosine computed within each participant.'
                        if metric == 'uncentered_spatial_cosine' else
                        'RMS computed within each participant after equal averaging of all four run maps; RMS has no sign.\n'
                        'BOLD is percent-scaled; gain/loss use monetary units. Coefficients depend on the fixed SPM impulse basis (oversampling 50), not peak response amplitude.')
        caption = ('Shaded row: S11 absolute coefficient map. Remaining rows: fixed specification differences.\n'
                   + measure_note + '\n'
                   + 'Points and bars copy saved participant means and uncorrected t95 intervals; no new inference.\n'
                   + mask_note + quality_note)
        figure.text(.02, .025, caption, fontsize=8.5, va='bottom', linespacing=1.5)
        figure.subplots_adjust(left=.27, right=.985, top=.84, bottom=.22, wspace=.12)
        for extension in ('png', 'pdf'):
            path = output/f'exploratory_fmri_{basename}.{extension}'
            figure.savefig(path, dpi=180, bbox_inches='tight')
            files.append(path.name)
        plt.close(figure)
    manifest = {
        'status': 'complete', 'source_summary': str(summary.resolve()),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'figures': files,
        'participant_counts': COUNTS, 'fixed_axes': dict(AXES),
        'plotted_rows': [rows[key] for key in sorted(rows)],
        'spatial_support': support, 'failed_run_arm_combinations': failed,
        'interpretation': 'Reporting only: saved participant means and t95 intervals, all six fixed axes, two slopes, and all/EI/ER strata. No fits, map reads, recomputed metrics, resampling, corrected tests, mechanism claims or original equivalence decisions.'}
    (output/'figure_generation.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require_compute()
    def timeout(signum, frame):
        raise TimeoutError('Reporting plot exceeded its 300-second bound')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(300)
    try:
        result = render(args.summary, args.output_dir)
        print(json.dumps({'status': result['status'], 'figures': result['figures']}), flush=True)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
