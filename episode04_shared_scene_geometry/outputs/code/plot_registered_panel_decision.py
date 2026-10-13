#!/usr/bin/env python3
"""Render the one saved EP04 decision; no score calculation or selection."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import tempfile
import textwrap
import time


EPISODE = Path(__file__).resolve().parents[2]
DECISION = EPISODE / 'outputs/runs/panel_decision/registered_panel_decision.json'
OUTPUT = EPISODE / 'outputs/figures/registered_panel_decision/20261005-full01'
PARTICIPANTS = ['sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07']
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
GATE_LABELS = {
    'practical_gain': 'Mean gain\n≥ 0.02',
    'participant_direction': 'Positive gain\nin ≥ 4/5 people',
    'participant_influence': 'Every person\nleave-one-out > 0',
    'sector_influence': 'Every ROI\nleave-one-out > 0',
    'positive_alignment': 'Alignment\n> 0',
    'image_null': 'Alignment >\nimage null',
    'geometry_null': 'Alignment >\ngeometry null',
    'low_level_partial_rsa': 'Low-level\npartial RSA',
    'near_neighbour_exclusion': 'Near-neighbour\nexclusion',
    'equal_source_strata': 'Equal source\nstrata',
    'complete_repeats': 'Complete\nrepeats',
    'neural_cosine': 'Neural\ncosine',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_vector(value, length, name):
    require(isinstance(value, list) and len(value) == length
            and all(type(item) in (int, float) and math.isfinite(item) for item in value),
            f'Complete finite saved {name} required')


def admit(report):
    require(isinstance(report, dict) and report.get('complete_admission') is True
            and type(report.get('candidate_decisions')) is int and report['candidate_decisions'] == 1
            and report.get('analysis_id') == 'candidate_decision'
            and report.get('mode') == 'complete_registered_fixed_panel_screen',
            'Only the one fully admitted saved decision may be plotted')
    require(report.get('status') in ('closed_no_candidate', 'selected_candidate_pending_exact_full_rerun')
            and report.get('candidate_selection_performed') is True
            and all(report.get(key) is False for key in ('configuration_lock_authorized',
                'candidate_lock_authorized', 'OOD_access_authorized', 'OOD_outcomes_opened',
                'exact_full_development_rerun_performed')),
            'The figure cannot imply lock, rerun completion, or OOD authority')
    models = report.get('VLM_order')
    require(isinstance(models, list) and len(models) == 3
            and all(isinstance(model, str) and model for model in models) and len(set(models)) == 3
            and report.get('participant_order') == PARTICIPANTS
            and report.get('fold_order') == list(range(5)),
            'All three saved models and all five frozen participants/folds required')
    effects = report['effects']
    numeric_vector(effects['gain'], 3, 'mean gains')
    numeric_vector(effects['unsubtracted_alignment'], 3, 'unsubtracted alignments')
    require(isinstance(effects['participant'], list) and len(effects['participant']) == 3,
            'All three models must have saved participant gains')
    for values in effects['participant']:
        numeric_vector(values, 5, 'participant gains')
    numeric_vector([report['image_permutation_threshold'], report['random_geometry_threshold']],
                   2, 'null thresholds')
    require(report.get('null_quantile_method') == 'higher'
            and report.get('null_exceedance') == 'strictly_greater_than',
            'Saved threshold interpretation differs from the fixed decision')
    gates = report['gates']
    require(isinstance(gates, list) and len(gates) == 3
            and all(isinstance(row, dict) and set(row) == set(GATE_LABELS)
                    and all(type(value) is bool for value in row.values()) for row in gates),
            'Every saved gate for each of the three models must be present')


def caption(report, matplotlib_version):
    return f"""# EP04: complete registered development-panel decision

Recorded status: `{report['status']}`. This figure preserves the saved decision;
it does not select, promote or lock a candidate. An indicated candidate remains
pending its registered exact full development rerun. OOD access and outcomes
remain closed.

**A. Native gain.** All five participant gains for each of the three VLMs are
the saved native RSA VLM-minus-DINOv2 Fisher-z contrasts, averaged over that
participant's five folds and three primary ROIs. White diamonds show the saved
equal-participant mean (`effects.gain`), without recomputing it. Reference lines
mark zero and the registered practical mean-gain threshold of 0.02. Individual
participants are not required to reach 0.02. No uncertainty intervals are shown.

**B. Alignment and null thresholds.** Points show saved *unsubtracted* VLM
alignment, averaged equally across participants, folds and primary ROIs. These
are alignment values, not VLM-minus-DINOv2 gains. The two lines separately show
the stored familywise image-permutation 95th percentile
({report['image_permutation_threshold']!r}) and random-geometry 95th percentile
({report['random_geometry_threshold']!r}). The registered quantile uses `higher`
and requires strict exceedance. Neither distribution nor threshold is recomputed.

**C. Complete gates.** All twelve saved Boolean gates are shown for every model.
PASS/FAIL is copied from the decision, including the five sensitivity gates;
the plot does not reevaluate or combine them into a new decision.

This is selection-exposed development evidence from five participants, not
population confirmation. All models and participants remain visible regardless
of the recorded outcome. The figure supplies no OOD result or new inference.

Sole scientific input: `{DECISION}`.
Rendering: Matplotlib {matplotlib_version}; no source arrays or other reports read.
"""


def render(report, check, plt, ListedColormap):
    models = report['VLM_order']
    names = ['\n'.join(textwrap.wrap(name, width=18)) for name in models]
    effects = report['effects']
    positions = [0, 1, 2]
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6B6B22']
    markers = ['o', '^', 's', 'P', 'X']
    offsets = [-0.24, -0.12, 0, 0.12, 0.24]
    fig = plt.figure(figsize=(17, 11), layout='constrained', facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 0.8])
    ax_gain, ax_alignment = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    ax_gates = fig.add_subplot(grid[1, :])
    for index, person in enumerate(PARTICIPANTS):
        ax_gain.scatter([position + offsets[index] for position in positions],
                        [row[index] for row in effects['participant']],
                        s=65, color=colors[index], marker=markers[index], label=person, zorder=3)
    ax_gain.scatter(positions, effects['gain'], s=105, marker='D', facecolors='white',
                    edgecolors='#18232D', linewidths=1.8, label='Saved mean', zorder=4)
    ax_gain.axhline(0, color='#59636C', linewidth=1, label='Zero gain')
    ax_gain.axhline(0.02, color='#925716', linestyle='--', linewidth=1.5,
                   label='Practical mean gain: 0.02')
    ax_gain.set(title='A. Native gain for every participant',
                ylabel='VLM − DINOv2 native RSA gain (Fisher z)', xticks=positions, xticklabels=names,
                xlim=(-0.5, 2.5))
    ax_gain.legend(fontsize=8.5, ncol=2, loc='best', framealpha=0.95)

    ax_alignment.vlines(positions, 0, effects['unsubtracted_alignment'], color='#688396', linewidth=2)
    ax_alignment.scatter(positions, effects['unsubtracted_alignment'], s=90, color='#244E68',
                         label='Saved VLM alignment', zorder=3)
    ax_alignment.axhline(0, color='#59636C', linewidth=0.8)
    ax_alignment.axhline(report['image_permutation_threshold'], color='#C16B12',
                         linestyle='--', linewidth=1.8, label='Image permutation: stored 95th percentile')
    ax_alignment.axhline(report['random_geometry_threshold'], color='#77589D',
                         linestyle=':', linewidth=2.2, label='Random geometry: stored 95th percentile')
    ax_alignment.set(title='B. Unsubtracted alignment and familywise null thresholds',
                     ylabel='Unsubtracted native alignment (Fisher z)', xticks=positions,
                     xticklabels=names, xlim=(-0.5, 2.5))
    ax_alignment.legend(fontsize=8.5, loc='best', framealpha=0.95)
    for axis in (ax_gain, ax_alignment):
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
        axis.margins(y=0.16)
    check()

    gate_names = list(GATE_LABELS)
    matrix = [[int(row[name]) for name in gate_names] for row in report['gates']]
    ax_gates.imshow(matrix, cmap=ListedColormap(['#F5D1C8', '#CCE7E2']), vmin=0, vmax=1, aspect='auto')
    for row in range(3):
        for column, name in enumerate(gate_names):
            ax_gates.text(column, row, 'PASS' if report['gates'][row][name] else 'FAIL',
                          ha='center', va='center', fontsize=10, weight='bold', color='#20323C')
    ax_gates.set_xticks(list(range(len(gate_names))), [GATE_LABELS[name] for name in gate_names], fontsize=9)
    ax_gates.set_yticks(positions, names, fontsize=10)
    ax_gates.set_xticks([index - 0.5 for index in range(len(gate_names) + 1)], minor=True)
    ax_gates.set_yticks([-0.5, 0.5, 1.5, 2.5], minor=True)
    ax_gates.grid(which='minor', color='white', linewidth=2)
    ax_gates.tick_params(which='both', length=0)
    ax_gates.set_title('C. Every registered gate for every VLM — saved PASS/FAIL', loc='left', pad=14)
    fig.suptitle('EP04 · complete registered development panel\n'
                 f"Saved status: {report['status']}\n"
                 'All three VLMs · five participants · selection-exposed development · OOD closed',
                 fontsize=14)
    return fig


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    require(0 < args.max_seconds <= 120, 'Explicit internal cap must be at most 120 seconds')
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in socket.gethostname()
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and os.environ.get('SLURM_JOB_CPUS_PER_NODE') == '1'
            and os.environ.get('SLURM_NNODES') == '1'
            and all(os.environ.get(name) == '1' for name in THREADS),
            'Actual one-CPU/one-node allocation and six thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE, 'Use the exact EP04 cwd')
    require(not DECISION.is_symlink() and DECISION.resolve() == DECISION,
            'The sole fixed decision input must not redirect')
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'Fresh fixed figure directory required; no overwrite')
    deadline = started + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Reporting figure deadline reached')

    def interrupted(*_):
        raise TimeoutError('Reporting figure interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
        # This is the only scientific input opened by the plotter.
        with DECISION.open() as stream:
            report = json.load(stream)
        admit(report)
        check()
        with tempfile.TemporaryDirectory(prefix='ep04-decision-plot-',
                                         dir=os.environ.get('L_SCRATCH_JOB')) as cache:
            os.environ['MPLCONFIGDIR'] = cache
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            fig = render(report, check, plt, ListedColormap)
            try:
                check()
                OUTPUT.mkdir(parents=True)
                for extension in ('png', 'svg'):
                    check()
                    with (OUTPUT / f'registered_panel_decision.{extension}').open('xb') as stream:
                        fig.savefig(stream, format=extension, dpi=180, facecolor='white')
                        stream.flush()
                        os.fsync(stream.fileno())
                check()
                # Caption is written last: a failed render retains explicit partial files.
                with (OUTPUT / 'caption.md').open('x') as stream:
                    stream.write(caption(report, matplotlib.__version__))
                    stream.flush()
                    os.fsync(stream.fileno())
                check()
            finally:
                plt.close(fig)
        print(json.dumps({'status': 'saved_decision_figure_written', 'output': str(OUTPUT),
                          'source_status': report['status'], 'elapsed_seconds': time.monotonic() - started,
                          'score_recomputation': False, 'candidate_decisions_created': 0,
                          'OOD_accessed': False}), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({'status': 'figure_not_completed', 'type': type(exc).__name__,
                          'message': str(exc), 'output': str(OUTPUT),
                          'existing_or_partial_files_preserved': True}), flush=True)
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
