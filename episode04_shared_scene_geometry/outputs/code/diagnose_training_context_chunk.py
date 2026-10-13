#!/usr/bin/env python3
"""Time one fixed sub-01 training-context chunk without exporting its values.

Run from EP04 in one coordinator-owned CPU/4 GiB step. The whole process has a
180-second internal deadline; the coordinator supplies a 190-second external
timeout plus ten seconds of termination grace. No source, fold, ROI, chunk or
time overrides are accepted. This is a cost/localization diagnostic, not a
completed reliability/NC analysis or authority to continue that analysis.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
PREPROCESSING_REPORT = Path(
    'outputs/runs/neural_preprocessing/20261003-remaining-folds01/sub-01/preprocessing.json'
)
METADATA_DIRECTORY = Path('outputs/runs/source_readiness/20261003-metadata_folds04')
INTERNAL_SECONDS = 180
CHUNK_COLUMNS = 256
THREAD_ENV = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)


def main():
    started = time.monotonic()
    cpu_started = time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    hostname = socket.gethostname()
    job_id = os.environ.get('SLURM_JOB_ID')
    if (not job_id or '-ln' in hostname
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(key) != '1' for key in THREAD_ENV)):
        raise RuntimeError('One actual compute CPU and all six numerical thread limits=1 required')
    if Path.cwd().resolve() != EPISODE:
        raise ValueError('Run from the exact EP04 directory')
    output = args.output_dir.resolve()
    output_root = EPISODE / 'outputs/runs/training_neural_context'
    if output.exists() or not output.is_relative_to(output_root):
        raise ValueError('Use a fresh EP04 training_neural_context output directory')
    output.mkdir(parents=True)
    report = {
        'analysis_id': 'training_neural_context', 'mode': 'single_chunk_diagnostic',
        'participant': 'sub-01', 'held_fold': 0, 'selected_columns': CHUNK_COLUMNS,
        'job_id': job_id, 'hostname': hostname, 'allocated_cpus': 1,
        'thread_environment': {key: os.environ[key] for key in THREAD_ENV},
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'internal_seconds': INTERNAL_SECONDS,
        'preprocessing_report': str(PREPROCESSING_REPORT),
        'metadata_directory': str(METADATA_DIRECTORY),
        'status': 'started', 'chunk_complete': False, 'events': [], 'errors': [],
        'beta_reads_attempted': 0, 'beta_reads_returned': 0,
        'beta_values_returned': 0, 'numerical_beta_access_attempted': False,
        'moment_calls_attempted': 0, 'moment_calls_returned': 0,
        'complete_training_context_components': False,
        'scientific_values_exported': False, 'full_context_continuation_launched': False,
        'training_parameters_refitted': False, 'primary_coverage_changed': False,
        'raw_images_embeddings_or_excluded_neural_outcomes_opened': False,
        'automatic_retry': False,
    }

    def check():
        if time.monotonic() - started >= INTERNAL_SECONDS:
            raise TimeoutError('Whole-process 180-second diagnostic deadline reached')

    def mark(phase, **details):
        check()
        event = {'phase': phase, 'utc': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': time.monotonic() - started,
                 'process_cpu_seconds': time.process_time() - cpu_started, **details}
        report['events'].append(event)
        print(json.dumps(event, allow_nan=False), flush=True)
        check()

    def interrupted(signum, _frame):
        raise TimeoutError(f'Diagnostic interrupted by signal {signum}')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['process_cpu_seconds'] = time.process_time() - cpu_started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pending = output / 'chunk_diagnostic.inprogress.json'
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        pending.replace(output / 'chunk_diagnostic.json')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(max(1, math.ceil(INTERNAL_SECONDS - (time.monotonic() - started))))
    try:
        mark('before_existing_loader_import')
        import numpy as np
        import training_neural_context as context
        from preprocess_development_neural import load_completed_inputs, read_json_lines
        mark('after_existing_loader_import')
        if context.CHUNK_VOXELS != CHUNK_COLUMNS:
            raise ValueError('The original 256-column chunk setting changed')

        mark('before_preprocessing_report')
        prepared = json.loads(PREPROCESSING_REPORT.read_text())
        folds = prepared.get('folds', [])
        if (prepared.get('participant') != 'sub-01'
                or prepared.get('complete_preprocessing') is not True
                or prepared.get('errors') != [] or len(folds) != 5
                or {fold.get('held_fold') for fold in folds} != set(range(5))
                or not all(fold.get('complete') is True for fold in folds)):
            raise ValueError('Exact completed sub-01 five-fold preprocessing required')
        mark('after_preprocessing_report')

        mark('before_completed_handoff_loader')
        images, sessions, voxel_indices, membership = load_completed_inputs(
            'sub-01', Path(prepared['participant_handoff']), METADATA_DIRECTORY)
        mark('after_completed_handoff_loader', images=len(images), sessions=len(sessions),
             voxels=len(voxel_indices))
        payload = Path(prepared['payload_dir'])
        mark('before_saved_preprocessing_identity')
        if (read_json_lines(payload / 'image_rows.jsonl') != images
                or not np.array_equal(np.load(payload / 'voxel_indices_c.npy', allow_pickle=False), voxel_indices)
                or not np.array_equal(np.load(payload / 'source_roi_membership.npy', allow_pickle=False), membership)):
            raise ValueError('Preprocessing and original handoff identities differ')
        image_folds = np.array([row['fold'] for row in images])
        mark('after_saved_preprocessing_identity')

        fold_source = payload / 'fold-0'
        mark('before_saved_training_parameters')
        with np.load(fold_source / 'training_parameters.npz', allow_pickle=False) as parameters:
            means, scales, support = (parameters[key] for key in ('mean', 'scale', 'retained_support'))
            estimable = bool(parameters['statistics_estimable'])
            if (parameters['session_names'].tolist() != list(context.SESSIONS)
                    or means.shape != (len(sessions), len(voxel_indices)) or scales.shape != means.shape
                    or support.shape != membership.shape or support.dtype != np.dtype(bool)
                    or np.any(support & ~membership)):
                raise ValueError('Saved parameters do not match original identities')
        mark('after_saved_training_parameters')
        mark('before_saved_repeat_support')
        with np.load(fold_source / 'repeat_support.npz', allow_pickle=False) as repeats:
            primary_counts, evc_counts, primary_eligible, evc_eligible = (
                repeats[key] for key in ('primary_counts', 'evc_counts', 'primary_eligible',
                                         'evc_eligible_on_primary_images'))
        mark('after_saved_repeat_support')
        mark('before_saved_presentation_validity')
        primary_valid, evc_valid = context.saved_validity(
            fold_source / 'presentation_validity.jsonl', sessions)
        mark('after_saved_presentation_validity')
        mark('before_chronological_repeat_assignment')
        common = context.common_training_images(image_folds, 0, primary_eligible, evc_eligible)
        selected_indices = np.flatnonzero(common)
        plans = [context.chronological_repeats(sessions, valid, common, 0)
                 for valid in (primary_valid, evc_valid)]
        for (_, counts), saved in zip(plans, (primary_counts, evc_counts)):
            if not np.array_equal(counts, saved[common]) or np.any(counts < 2):
                raise ValueError('Saved repeat support differs from original assignments')
        records, _counts = plans[0]
        columns = np.flatnonzero(support[:, 0])[:CHUNK_COLUMNS]
        if len(columns) != CHUNK_COLUMNS or not len(records) or not estimable:
            raise ValueError('Selected first ROI lacks the fixed estimable 256-column training chunk')
        for session_index, session in enumerate(sessions):
            rows = records[records[:, 0] == session_index, 1]
            if np.any(session.folds[rows] == 0):
                raise ValueError('Held-out rows cannot enter the selected training chunk')
        report.update(roi=context.ROI_NAMES[0], common_training_images=len(selected_indices),
                      training_presentations=len(records), chunk_columns=len(columns))
        mark('after_chronological_repeat_assignment', roi=context.ROI_NAMES[0],
             common_training_images=len(selected_indices), training_presentations=len(records),
             chunk_columns=len(columns))

        class TimedBeta:
            """Forward the original memmap indexing and its returned object unchanged."""
            def __init__(self, session):
                self.session = session

            def __getitem__(self, index):
                check()
                rows, selected = index
                report['beta_reads_attempted'] += 1
                report['numerical_beta_access_attempted'] = True
                mark('before_beta_slice', session=self.session.name,
                     rows=int(rows.size), columns=int(selected.size))
                returned = self.session.betas[index]
                report['beta_reads_returned'] += 1
                report['beta_values_returned'] += int(returned.size)
                mark('after_beta_slice', session=self.session.name,
                     rows=int(rows.size), columns=int(selected.size))
                return returned

        original_moments = context.repeat_moments

        def timed_moments(values, image_indices, groups, n_images):
            report['moment_calls_attempted'] += 1
            mark('before_repeat_moments', presentations=int(values.shape[0]),
                 columns=int(values.shape[1]), images=int(n_images))
            returned = original_moments(values, image_indices, groups, n_images)
            report['moment_calls_returned'] += 1
            mark('after_repeat_moments')
            return returned

        timed_sessions = [replace(session, betas=TimedBeta(session)) for session in sessions]
        mark('before_original_roi_context')
        # Process-local timing only: both existing scientific functions and all
        # their operands are unchanged. Restore the binding on every exit.
        context.repeat_moments = timed_moments
        try:
            result = context.roi_context(timed_sessions, means, scales, columns,
                                         records, len(selected_indices))
        finally:
            context.repeat_moments = original_moments
        del result
        mark('after_original_roi_context')
        if report['moment_calls_returned'] != 1:
            raise ValueError('Exactly one original chunk moment call was selected')
        check()
        report.update(status='single_chunk_diagnostic_completed', chunk_complete=True)
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report.update(status='single_chunk_diagnostic_incomplete', chunk_complete=False)
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() - started >= INTERNAL_SECONDS:
            report.update(status='single_chunk_diagnostic_incomplete', chunk_complete=False)
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['chunk_complete'] and time.monotonic() - started >= INTERNAL_SECONDS:
            report.update(status='single_chunk_diagnostic_incomplete', chunk_complete=False)
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['chunk_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
