#!/usr/bin/env python3
"""Display the complete, fixed five-person descriptive training-context table.

Reads only the five named completed JSON reports. Copies every saved row to
CSV; dots summarize saved Fisher z across five overlapping training folds.
No raw arrays, selection, contrasts, inference, nulls or subsequent dispatch.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
PARTICIPANTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
MODELS = ('CLIP', 'DINOv2', 'PEcore', 'SigLIP2')
ROIS = ('laion-ventral', 'laion-lateral', 'laion-dorsal', 'retinotopic-EVC')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
RUNS = EPISODE / 'outputs/runs/training_context_rsa'
SOURCES = {
    'sub-01': RUNS / '20261005-rsa-pilot-46658777/sub-01/training_context_rsa.json',
    **{person: RUNS / '20261005-rsa-remaining-46660232' / person / 'training_context_rsa.json'
       for person in PARTICIPANTS[1:]},
}
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
CSV_FIELDS = (
    'participant', 'held_fold', 'roi', 'model_label', 'model_id',
    'training_images', 'retained_voxels', 'training_image_id_file',
    'context_per_voxel_artifact', 'valid_training_presentations',
    'repeat_count_histogram', 'repeat_reliability', 'provider_formula_NC_percent',
    'independent_repeat_errors_established', 'rsa_spearman_r', 'rsa_fisher_z',
    'status', 'source_report',
)
JSON_FIELDS = ('repeat_count_histogram', 'repeat_reliability', 'provider_formula_NC_percent')


def read_complete_report(participant):
    """Admit all required saved rows; never drop an undefined or missing unit."""
    path = SOURCES[participant]
    report = json.loads(path.read_text())
    expected_job = '46658777' if participant == 'sub-01' else '46660232'
    if (report.get('participant') != participant or str(report.get('job_id')) != expected_job
            or report.get('analysis_id') != 'training_neural_context'
            or report.get('mode') != 'matched_training_context_RSA'
            or report.get('status') != 'participant_training_context_RSA_completed'
            or report.get('complete_participant_training_context_RSA') is not True
            or report.get('all_required_rows_recorded') is not True
            or not report.get('finished_utc') or report.get('errors') != []):
        raise ValueError(f'Exact completed training-context report required: {participant}')
    closed_flags = ('primary_panel_evaluated', 'candidate_selection_performed',
                    'held_out_performance', 'raw_beta_arrays_read',
                    'raw_images_or_excluded_outcomes_opened',
                    'preprocessing_or_noise_ceiling_refitted')
    if any(report.get(key) is not False for key in closed_flags) or report.get('null_draws') != 0:
        raise ValueError('Saved report does not have the selected descriptive-only scope')
    folds = report.get('folds', [])
    if (len(folds) != 5 or {fold['held_fold'] for fold in folds} != set(range(5))
            or any(fold.get('complete') is not True or fold.get('rows_recorded') != 16
                   or fold.get('matched_image_and_repeat_support') is not True for fold in folds)):
        raise ValueError('Five complete matched-support folds required')
    rows = report.get('rows', [])
    expected = {(fold, roi, model) for fold in range(5) for roi in ROIS for model in MODELS}
    identities = [(row['held_fold'], row['roi'], row['model_label']) for row in rows]
    if len(rows) != 80 or len(set(identities)) != 80 or set(identities) != expected:
        raise ValueError('Exactly 80 unique, fixed fold/ROI/model rows required')
    by_fold = {fold['held_fold']: fold for fold in folds}
    for row in rows:
        if (row['participant'] != participant or row.get('status') != 'evaluable'
                or row.get('independent_repeat_errors_established') is not False
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in ('rsa_spearman_r', 'rsa_fisher_z'))
                or not -1 < row['rsa_spearman_r'] < 1):
            raise ValueError('Every saved row must be evaluable and finite')
        fold = by_fold[row['held_fold']]
        if (row['training_images'] != fold['training_images']
                or row['training_image_id_file'] != fold['training_image_id_file']
                or row['training_images'] < 3 or row['retained_voxels'] < 3):
            raise ValueError('Saved row support differs from its completed fold')
        for key in JSON_FIELDS:
            if not isinstance(row[key], dict):
                raise ValueError('Saved repeat/reliability/NC context must be retained')
    return rows


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname()
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(key) != '1' for key in THREAD_ENV)):
        raise RuntimeError('Use one actual compute CPU and all six numerical thread limits=1')
    output = args.output_dir.resolve()
    if Path.cwd().resolve() != EPISODE or output.exists() or not output.is_relative_to(EPISODE / 'outputs/figures'):
        raise ValueError('Use EP04 cwd and a fresh episode-local figures directory')
    output.mkdir(parents=True)
    report = {'status': 'started', 'completed': False, 'saved_rows': 0,
              'plotted_participant_summaries': 0, 'errors': [],
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'job_id': os.environ['SLURM_JOB_ID'], 'internal_seconds': 90,
              'source_reports': {person: str(path) for person, path in SOURCES.items()},
              'interpretation': 'Descriptive training data; not held-out performance',
              'primary_panel_evaluated': False, 'candidate_selection_performed': False,
              'raw_data_read': False, 'new_null_or_RDM_or_RSA_computation': False}

    def check():
        if time.monotonic() - started >= 90:
            raise TimeoutError('Display whole-process deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        pending = output / 'display.inprogress.json'
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        pending.replace(output / 'display.json')

    def interrupted(signum, _frame):
        raise TimeoutError(f'Display interrupted by signal {signum}')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(max(1, math.ceil(90 - (time.monotonic() - started))))
    try:
        check()
        all_rows = []
        for participant in PARTICIPANTS:
            check()
            all_rows.extend(dict(row, source_report=str(SOURCES[participant]))
                            for row in read_complete_report(participant))
        if len(all_rows) != 400:
            raise ValueError('All 400 saved rows are required before display')
        # Fixed display order, never a sort by performance.
        rows_by_key = {(r['participant'], r['held_fold'], r['roi'], r['model_label']): r for r in all_rows}
        csv_path = output / 'training_context_rsa.csv'
        csv_pending = output / 'training_context_rsa.inprogress.csv'
        with csv_pending.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction='raise')
            writer.writeheader()
            for participant in PARTICIPANTS:
                for fold in range(5):
                    for roi in ROIS:
                        for model in MODELS:
                            original = rows_by_key[(participant, fold, roi, model)]
                            row = {key: original[key] for key in CSV_FIELDS}
                            for key in JSON_FIELDS:
                                row[key] = json.dumps(row[key], sort_keys=True, allow_nan=False)
                            writer.writerow(row)
            stream.flush()
        check()
        csv_pending.replace(csv_path)
        report['saved_rows'] = 400
        save()

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        values = {}
        for roi in ROIS:
            for model in MODELS:
                for participant in PARTICIPANTS:
                    z = [rows_by_key[(participant, fold, roi, model)]['rsa_fisher_z'] for fold in range(5)]
                    values[(roi, model, participant)] = math.tanh(math.fsum(z) / 5)
        check()
        fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
        try:
            offsets = np.linspace(-0.16, 0.16, len(PARTICIPANTS))
            low = min(0.0, min(values.values()) - 0.02)
            high = max(values.values()) + 0.02
            for ax, roi in zip(axes.flat, ROIS):
                for p_index, (participant, color) in enumerate(zip(PARTICIPANTS, COLORS)):
                    y = [values[(roi, model, participant)] for model in MODELS]
                    ax.scatter(np.arange(4) + offsets[p_index], y, s=48,
                               color=color, edgecolor='white', linewidth=0.5,
                               label=participant, zorder=3)
                ax.axhline(0, color='0.55', linewidth=0.7)
                ax.set_title(roi)
                ax.set_xticks(np.arange(4), MODELS)
                ax.set_xlim(-0.45, 3.45)
                ax.set_ylim(low, high)
                ax.set_ylabel('tanh(mean of five saved Fisher z values)')
                ax.grid(axis='y', color='0.9', linewidth=0.6)
                ax.spines[['top', 'right']].set_visible(False)
            fig.suptitle('Matched training-context RSA', fontsize=17, y=0.97)
            fig.text(0.5, 0.92, 'Descriptive training data; not held-out performance',
                     ha='center', fontsize=13, fontweight='bold')
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.115),
                       ncol=5, frameon=False, title='Participant')
            fig.text(0.5, 0.063,
                     'Each dot summarizes one participant across five overlapping training folds.\n'
                     'Participants have different image pools; dots are not matched-image replications.\n'
                     'Fixed model order; no winner selection, contrasts, nulls or uncertainty intervals.',
                     ha='center', va='center', fontsize=10)
            fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.23,
                                hspace=0.35, wspace=0.23)
            for extension in ('png', 'svg'):
                check()
                pending = output / f'training_context_rsa.inprogress.{extension}'
                fig.savefig(pending, format=extension, dpi=160, facecolor='white')
                check()
                pending.replace(output / f'training_context_rsa.{extension}')
        finally:
            plt.close(fig)
        check()
        report.update(status='complete', completed=True, plotted_participant_summaries=80,
                      csv=str(csv_path), png=str(output / 'training_context_rsa.png'),
                      svg=str(output / 'training_context_rsa.svg'))
    except Exception as exc:
        report.update(status='incomplete', completed=False)
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() - started >= 90:
            report.update(status='incomplete', completed=False)
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['completed'] and time.monotonic() - started >= 90:
            report.update(status='incomplete', completed=False)
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    print(json.dumps({'status': report['status'], 'saved_rows': report['saved_rows'],
                      'plotted_participant_summaries': report['plotted_participant_summaries'],
                      'output_dir': str(output)}, allow_nan=False), flush=True)
    return 0 if report['completed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
