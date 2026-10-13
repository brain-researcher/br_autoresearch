#!/usr/bin/env python3
"""Complete the fixed sub-01 five-fold context with original chunk arithmetic.

This operational continuation retains the original scientific helpers, all
four ROIs, saved training parameters and 256-column order. It has no resume,
retry, participant, fold, source or scientific-setting overrides. Use one
actual CPU/4 GiB, a 2400-second internal cap and a coordinator-supplied
2460-second external timeout plus ten seconds of termination grace.
"""
import argparse
from collections import Counter
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
INTERNAL_SECONDS = 2400
THREAD_ENV = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)


def main():
    started = time.monotonic()
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
        'analysis_id': 'training_neural_context', 'participant': 'sub-01',
        'phase': 'fixed_sub01_full_context_continuation',
        'job_id': job_id, 'hostname': hostname, 'allocated_cpus': 1,
        'thread_environment': {key: os.environ[key] for key in THREAD_ENV},
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'internal_max_seconds': INTERNAL_SECONDS,
        'preprocessing_report': str(PREPROCESSING_REPORT),
        'metadata_directory': str(METADATA_DIRECTORY),
        'complete_training_context_components': False,
        'complete_RSA_context_diagnostic': False,
        'scope': 'Training-only repeat reliability/provider-formula NC; diagnostic intersection only',
        'primary_image_coverage_changed': False, 'saved_parameters_refitted': False,
        'mixed_source_released_NC_raw_pixels_or_embeddings_opened': False,
        'RDM_or_RSA_computed': False, 'automatic_retry': False, 'automatic_resume': False,
        'status': 'started', 'folds': [], 'errors': [],
        'completed_chunks': 0,
        'partial_array_rule': 'Only the reported leading completed_voxels are available; trailing NaNs are uncomputed',
    }

    def check():
        if time.monotonic() - started >= INTERNAL_SECONDS:
            raise TimeoutError('Whole-process 2400-second continuation deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pending = output / 'training_context.inprogress.json'
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        pending.replace(output / 'training_context.json')

    def progress(stage, **details):
        check()
        report['operational_progress'] = {
            'stage': stage, 'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - started, **details,
        }
        save()
        print(json.dumps(report['operational_progress'], allow_nan=False), flush=True)
        check()

    def interrupted(signum, _frame):
        raise TimeoutError(f'Continuation interrupted by signal {signum}')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(max(1, math.ceil(INTERNAL_SECONDS - (time.monotonic() - started))))
    try:
        progress('before_original_helper_import')
        import numpy as np
        import training_neural_context as original
        if (original.CHUNK_VOXELS != 256 or tuple(original.FOLDS) != tuple(range(5))
                or tuple(original.ROI_NAMES) !=
                ('laion-ventral', 'laion-lateral', 'laion-dorsal', 'retinotopic-EVC')):
            raise ValueError('The selected original fold/ROI/chunk identity changed')
        progress('before_completed_preprocessing_report')
        prepared = json.loads(PREPROCESSING_REPORT.read_text())
        if (prepared.get('participant') != 'sub-01'
                or prepared.get('complete_preprocessing') is not True
                or prepared.get('errors') != [] or len(prepared.get('folds', [])) != 5
                or {fold.get('held_fold') for fold in prepared['folds']} != set(original.FOLDS)
                or not all(fold.get('complete') is True for fold in prepared['folds'])):
            raise ValueError('Completed sub-01 five-fold preprocessing required')
        progress('before_completed_handoff_loader')
        images, sessions, voxel_indices, membership = original.load_completed_inputs(
            'sub-01', Path(prepared['participant_handoff']), METADATA_DIRECTORY)
        payload = Path(prepared['payload_dir'])
        progress('before_preprocessing_identity')
        if (original.read_json_lines(payload / 'image_rows.jsonl') != images
                or not np.array_equal(np.load(payload / 'voxel_indices_c.npy', allow_pickle=False), voxel_indices)
                or not np.array_equal(np.load(payload / 'source_roi_membership.npy', allow_pickle=False), membership)):
            raise ValueError('Preprocessing and source handoff image/voxel identities differ')
        image_folds = np.array([row['fold'] for row in images])

        for held_fold in original.FOLDS:
            progress('before_fold_parameters', held_fold=held_fold)
            fold_source = payload / f'fold-{held_fold}'
            with np.load(fold_source / 'training_parameters.npz', allow_pickle=False) as parameters:
                means, scales, support = (parameters[key] for key in ('mean', 'scale', 'retained_support'))
                estimable = bool(parameters['statistics_estimable'])
                if (parameters['session_names'].tolist() != list(original.SESSIONS)
                        or means.shape != (len(sessions), len(voxel_indices)) or scales.shape != means.shape
                        or support.shape != membership.shape or support.dtype != np.dtype(bool)
                        or np.any(support & ~membership)):
                    raise ValueError('Saved parameters do not match fixed session/voxel/ROI identities')
            progress('before_fold_repeat_support', held_fold=held_fold)
            with np.load(fold_source / 'repeat_support.npz', allow_pickle=False) as repeats:
                primary_counts, evc_counts, primary_eligible, evc_eligible = (
                    repeats[key] for key in ('primary_counts', 'evc_counts', 'primary_eligible',
                                             'evc_eligible_on_primary_images'))
            progress('before_fold_validity', held_fold=held_fold)
            primary_valid, evc_valid = original.saved_validity(
                fold_source / 'presentation_validity.jsonl', sessions)
            common = original.common_training_images(image_folds, held_fold, primary_eligible, evc_eligible)
            selected_indices = np.flatnonzero(common)
            progress('before_fold_repeat_assignment', held_fold=held_fold)
            plans = [original.chronological_repeats(sessions, valid, common, held_fold)
                     for valid in (primary_valid, evc_valid)]
            for (_, counts), saved in zip(plans, (primary_counts, evc_counts)):
                if not np.array_equal(counts, saved[common]) or np.any(counts < 2):
                    raise ValueError('Saved image and presentation repeat support disagree')
            fold_dir = output / f'fold-{held_fold}'
            fold_dir.mkdir()
            fold_report = {
                'held_fold': held_fold, 'complete': False,
                'training_statistics_estimable': estimable,
                'common_training_images': len(selected_indices),
                'primary_training_images': int(np.count_nonzero((image_folds != held_fold) & primary_eligible)),
                'primary_training_images_missing_EVC_support': int(np.count_nonzero(
                    (image_folds != held_fold) & primary_eligible & ~evc_eligible)),
                'primary_evaluation_images_missing_EVC_support': int(np.count_nonzero(
                    (image_folds == held_fold) & primary_eligible & ~evc_eligible)),
                'image_support': str(fold_dir / 'training_image_support.jsonl'),
                'repeat_assignments': str(fold_dir / 'repeat_assignments.npz'), 'rois': {},
            }
            report['folds'].append(fold_report)
            progress('before_fold_identity_publication', held_fold=held_fold)
            with (fold_dir / 'training_image_support.jsonl').open('x') as stream:
                for index in np.flatnonzero(image_folds != held_fold):
                    stream.write(json.dumps({
                        'image_id': images[index]['image_id'], 'frozen_image_index': int(index),
                        'primary_valid_repeats': int(primary_counts[index]),
                        'EVC_valid_repeats': int(evc_counts[index]),
                        'primary_eligible': bool(primary_eligible[index]),
                        'common_context_image': bool(common[index]),
                        'context_exclusion_reason': None if common[index] else (
                            'primary_ineligible' if not primary_eligible[index] else 'EVC_missing_minimum_two'),
                    }) + '\n')
            np.savez(fold_dir / 'repeat_assignments.npz', common_frozen_image_indices=selected_indices,
                     columns=np.asarray(['session_index', 'session_row_index', 'common_image_index',
                                         'run', 'session_beta_axis_index', 'alternating_group']),
                     session_names=np.asarray(original.SESSIONS), primary=plans[0][0], EVC=plans[1][0])

            for roi, name in enumerate(original.ROI_NAMES):
                records, counts = plans[0 if roi < 3 else 1]
                columns = np.flatnonzero(support[:, roi])
                # Same result allocation and chunk order as original.roi_context.
                # Each column's arithmetic remains entirely in the old helper.
                results = [np.full(len(columns), np.nan) for _ in range(3)]
                partial_path = fold_dir / f'{name}_context.inprogress.npz'
                final_path = fold_dir / f'{name}_context.npz'
                roi_report = {'complete': False, 'retained_voxels': len(columns),
                              'completed_voxels': 0, 'completed_chunks': 0,
                              'partial_per_voxel_values': str(partial_path)}
                fold_report['rois'][name] = roi_report
                for start in range(0, len(columns), original.CHUNK_VOXELS):
                    stop = min(start + original.CHUNK_VOXELS, len(columns))
                    progress('before_original_chunk', held_fold=held_fold, roi=name,
                             start_column=start, stop_column=stop)
                    # No transform, fitted parameter, row, order or formula changes.
                    block_result = original.roi_context(
                        sessions, means, scales, columns[start:stop], records,
                        len(selected_indices), chunk=original.CHUNK_VOXELS)
                    check()
                    for result, computed in zip(results, block_result):
                        result[start:stop] = computed
                    pending = fold_dir / f'{name}_context.checkpoint.npz'
                    np.savez(pending, voxel_indices_c=voxel_indices[columns], eta=results[0],
                             provider_formula_NC_percent=results[1], signed_repeat_correlation=results[2])
                    pending.replace(partial_path)
                    roi_report['completed_voxels'] = stop
                    roi_report['completed_chunks'] += 1
                    report['completed_chunks'] += 1
                    progress('after_original_chunk', held_fold=held_fold, roi=name,
                             start_column=start, stop_column=stop)
                check()
                if not len(columns):
                    # Preserve the original helper's empty-support artifact.
                    np.savez(partial_path, voxel_indices_c=voxel_indices[columns], eta=results[0],
                             provider_formula_NC_percent=results[1], signed_repeat_correlation=results[2])
                if final_path.exists():
                    raise FileExistsError('Refusing to overwrite a completed ROI artifact')
                partial_path.replace(final_path)
                roi_report['per_voxel_values'] = str(final_path)
                roi_report.pop('partial_per_voxel_values')
                eta, nc, reliability = results
                image_sessions = np.unique(records[:, [2, 0]], axis=0)
                spanning = int(np.count_nonzero(np.bincount(
                    image_sessions[:, 0], minlength=len(selected_indices)) > 1))
                roi_report.update({
                    'common_training_images': len(selected_indices),
                    'repeat_assignment_group': 'primary' if roi < 3 else 'EVC',
                    'valid_training_presentations': len(records),
                    'repeat_count_histogram': dict(sorted(Counter(int(value) for value in counts).items())),
                    'h_mean_inverse_repeat_count': float(np.mean(1. / counts)) if len(counts) else None,
                    'presentations_by_session': {session: int(np.count_nonzero(records[:, 0] == index))
                                                 for index, session in enumerate(original.SESSIONS)},
                    'images_with_repeats_spanning_sessions': spanning,
                    'independent_repeat_errors_established': False,
                    'repeat_reliability': original.summary(reliability),
                    'provider_formula_NC_percent': original.summary(nc),
                    'zero_residual_noise_voxels': int(np.count_nonzero(eta == 0)),
                    'eta_at_least_one_voxels': int(np.count_nonzero(eta >= 1)),
                    'per_voxel_values': str(final_path),
                    'support_evaluable': bool(estimable and len(columns) >= 3 and len(selected_indices) >= 2),
                })
                check()
                roi_report['complete'] = True
                progress('roi_completed', held_fold=held_fold, roi=name)
            check()
            fold_report['complete'] = True
            progress('fold_completed', held_fold=held_fold)
        check()
        report['complete_training_context_components'] = len(report['folds']) == 5 and all(
            fold['complete'] for fold in report['folds'])
        report['status'] = 'training_context_components_completed'
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['complete_training_context_components'] = False
        report['status'] = 'training_context_components_incomplete'
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() - started >= INTERNAL_SECONDS:
            report['complete_training_context_components'] = False
            report['status'] = 'training_context_components_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_training_context_components'] and time.monotonic() - started >= INTERNAL_SECONDS:
            report['complete_training_context_components'] = False
            report['status'] = 'training_context_components_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_training_context_components'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
