#!/usr/bin/env python3
"""Training-only repeat reliability and provider-formula NC; no RSA or RDMs."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from preprocess_development_neural import (
    CHUNK_VOXELS, FOLDS, ROI_NAMES, SESSIONS, load_completed_inputs, read_json_lines,
)


def common_training_images(image_folds, held_fold, primary_eligible, evc_eligible):
    """Diagnostic intersection only; inputs and primary eligibility are unchanged."""
    return (image_folds != held_fold) & primary_eligible & evc_eligible


def chronological_repeats(sessions, valid, common, held_fold):
    """Assign all valid training repeats chronologically, alternating from group0."""
    selected_images = np.flatnonzero(common)
    local_image = np.full(len(common), -1, dtype=np.int64)
    local_image[selected_images] = np.arange(len(selected_images))
    records = []
    for session_index, (session, flags) in enumerate(zip(sessions, valid)):
        for row_index in np.flatnonzero(flags & common[session.image_indices]
                                        & (session.folds != held_fold)):
            row = session.rows[row_index]
            records.append((session.name, int(row['run']), int(row['session_beta_axis_index']),
                            session_index, int(row_index), int(local_image[session.image_indices[row_index]])))
    records.sort(key=lambda row: row[:3])
    counts = np.zeros(len(selected_images), dtype=np.int64)
    rows = []
    for _, run, axis, session_index, row_index, image_index in records:
        group = int(counts[image_index] % 2)
        counts[image_index] += 1
        rows.append((session_index, row_index, image_index, run, axis, group))
    columns = np.asarray(rows, dtype=np.int64).reshape(-1, 6)
    return columns, counts


def repeat_moments(values, image_indices, groups, n_images):
    """Equal-image residual variance, unequal-repeat NC and signed split r."""
    values = np.asarray(values, dtype=np.float64)
    n_voxels = values.shape[1]
    eta = np.full(n_voxels, np.nan)
    nc = np.full(n_voxels, np.nan)
    reliability = np.full(n_voxels, np.nan)
    counts = np.bincount(image_indices, minlength=n_images)
    if n_images == 0 or np.any(counts < 2):
        return eta, nc, reliability
    finite = np.isfinite(values).all(axis=0)
    if not finite.any():
        return eta, nc, reliability
    selected = values[:, finite]
    sums = np.zeros((n_images, selected.shape[1]), dtype=np.float64)
    np.add.at(sums, image_indices, selected)
    means = sums / counts[:, None]
    # Center within each image before squaring: no subtraction of raw moments.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        residual_squares = (selected - means[image_indices]) ** 2
        residual_sums = np.zeros_like(sums)
        np.add.at(residual_sums, image_indices, residual_squares)
        local_eta = (residual_sums / (counts[:, None] - 1)).mean(axis=0)
        local_eta[~np.isfinite(local_eta) | (local_eta < 0)] = np.nan
        signal_variance = np.maximum(0., 1. - local_eta)
        h = np.mean(1. / counts)
        local_nc = 100. * signal_variance / (signal_variance + local_eta * h)
    eta[finite], nc[finite] = local_eta, local_nc
    if n_images < 2:
        return eta, nc, reliability
    halves = []
    for group in (0, 1):
        group_mask = groups == group
        group_counts = np.bincount(image_indices[group_mask], minlength=n_images)
        if np.any(group_counts == 0):
            return eta, nc, reliability
        group_sum = np.zeros_like(sums)
        np.add.at(group_sum, image_indices[group_mask], selected[group_mask])
        half = group_sum / group_counts[:, None]
        halves.append(half - half.mean(axis=0))
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        numerator = np.sum(halves[0] * halves[1], axis=0)
        denominator = np.sqrt(np.sum(halves[0] ** 2, axis=0)) * np.sqrt(np.sum(halves[1] ** 2, axis=0))
        local_r = numerator / denominator
    local_r[~np.isfinite(local_r) | (denominator <= 0)] = np.nan
    reliability[finite] = local_r
    return eta, nc, reliability


def roi_context(sessions, means, scales, columns, records, n_images, chunk=CHUNK_VOXELS):
    """Read only selected training rows; use saved parameters without refitting."""
    results = [np.full(len(columns), np.nan) for _ in range(3)]
    for start in range(0, len(columns), chunk):
        selected_columns = columns[start:start + chunk]
        values = np.empty((len(records), len(selected_columns)), dtype=np.float64)
        for session_index, session in enumerate(sessions):
            positions = np.flatnonzero(records[:, 0] == session_index)
            if not len(positions):
                continue
            rows = records[positions, 1]
            block = np.asarray(session.betas[np.ix_(rows, selected_columns)], dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                values[positions] = ((block - means[session_index, selected_columns])
                                     / scales[session_index, selected_columns])
        block_result = repeat_moments(values, records[:, 2], records[:, 5], n_images)
        for result, computed in zip(results, block_result):
            result[start:start + len(selected_columns)] = computed
    return tuple(results)


def summary(values):
    defined = values[np.isfinite(values)]
    return {'defined_voxels': len(defined), 'undefined_voxels': len(values) - len(defined),
            'median': float(np.median(defined)) if len(defined) else None,
            'p10': float(np.quantile(defined, .1)) if len(defined) else None,
            'p90': float(np.quantile(defined, .9)) if len(defined) else None,
            'negative_voxels': int(np.count_nonzero(defined < 0))}


def saved_validity(path, sessions):
    rows = read_json_lines(path)
    flags = [[], []]
    offset = 0
    for session in sessions:
        selected = rows[offset:offset + len(session.rows)]
        if len(selected) != len(session.rows):
            raise ValueError('Saved presentation validity is incomplete')
        for actual, expected in zip(selected, session.rows):
            if any(actual[key] != expected[key] for key in ('session', 'image_id', 'session_beta_axis_index')):
                raise ValueError('Saved validity differs from the completed trial identities')
            if any(type(actual[key]) is not bool for key in ('primary_valid', 'evc_valid')):
                raise ValueError('Saved presentation validity must retain exact Boolean flags')
        for destination, key in zip(flags, ('primary_valid', 'evc_valid')):
            destination.append(np.array([row[key] for row in selected], dtype=bool))
        offset += len(selected)
    if offset != len(rows):
        raise ValueError('Unexpected rows in saved validity')
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preprocessing-report', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=840)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Verified coordinator-owned compute allocation required')
    if args.output_dir.exists() or not 0 < args.max_seconds <= 840:
        raise ValueError('Fresh output directory and positive deadline<=840s required')
    prepared = json.loads(args.preprocessing_report.read_text())
    if (prepared.get('complete_preprocessing') is not True or prepared.get('errors') != []
            or len(prepared.get('folds', [])) != 5
            or {fold.get('held_fold') for fold in prepared['folds']} != set(FOLDS)
            or not all(fold.get('complete') is True for fold in prepared['folds'])):
        raise ValueError('Completed five-fold preprocessing required')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {'analysis_id': 'training_neural_context', 'participant': prepared['participant'],
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'internal_max_seconds': args.max_seconds,
              'preprocessing_report': str(args.preprocessing_report),
              'complete_training_context_components': False, 'complete_RSA_context_diagnostic': False,
              'scope': 'Training-only repeat reliability/provider-formula NC; diagnostic intersection only',
              'primary_image_coverage_changed': False, 'saved_parameters_refitted': False,
              'mixed_source_released_NC_raw_pixels_or_embeddings_opened': False,
              'RDM_or_RSA_computed': False, 'folds': [], 'errors': []}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output_dir / 'training_context.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output_dir / 'training_context.json')
        print(json.dumps({'folds_recorded': len(report['folds']),
                          'elapsed_seconds': report['elapsed_seconds']}), flush=True)

    def deadline(*_):
        raise TimeoutError(f'Training-context internal{args.max_seconds}s ceiling reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        images, sessions, voxel_indices, membership = load_completed_inputs(
            prepared['participant'], Path(prepared['participant_handoff']), args.metadata_dir)
        payload = Path(prepared['payload_dir'])
        if (read_json_lines(payload / 'image_rows.jsonl') != images
                or not np.array_equal(np.load(payload / 'voxel_indices_c.npy', allow_pickle=False), voxel_indices)
                or not np.array_equal(np.load(payload / 'source_roi_membership.npy', allow_pickle=False), membership)):
            raise ValueError('Preprocessing and source handoff image/voxel identities differ')
        image_folds = np.array([row['fold'] for row in images])
        for held_fold in FOLDS:
            fold_source = payload / f'fold-{held_fold}'
            with np.load(fold_source / 'training_parameters.npz', allow_pickle=False) as parameters:
                means, scales, support = (parameters[key] for key in ('mean', 'scale', 'retained_support'))
                estimable = bool(parameters['statistics_estimable'])
                if (parameters['session_names'].tolist() != list(SESSIONS)
                        or means.shape != (len(sessions), len(voxel_indices)) or scales.shape != means.shape
                        or support.shape != membership.shape or support.dtype != np.dtype(bool)
                        or np.any(support & ~membership)):
                    raise ValueError('Saved parameters do not match fixed session/voxel/ROI identities')
            with np.load(fold_source / 'repeat_support.npz', allow_pickle=False) as repeats:
                primary_counts, evc_counts, primary_eligible, evc_eligible = (
                    repeats[key] for key in ('primary_counts', 'evc_counts', 'primary_eligible', 'evc_eligible_on_primary_images'))
            primary_valid, evc_valid = saved_validity(fold_source / 'presentation_validity.jsonl', sessions)
            common = common_training_images(image_folds, held_fold, primary_eligible, evc_eligible)
            selected_indices = np.flatnonzero(common)
            plans = [chronological_repeats(sessions, valid, common, held_fold)
                     for valid in (primary_valid, evc_valid)]
            for (_, counts), saved in zip(plans, (primary_counts, evc_counts)):
                if not np.array_equal(counts, saved[common]) or np.any(counts < 2):
                    raise ValueError('Saved image and presentation repeat support disagree')
            fold_dir = args.output_dir / f'fold-{held_fold}'
            fold_dir.mkdir()
            fold_report = {'held_fold': held_fold, 'complete': False,
                'training_statistics_estimable': estimable,
                'common_training_images': len(selected_indices),
                'primary_training_images': int(np.count_nonzero((image_folds != held_fold) & primary_eligible)),
                'primary_training_images_missing_EVC_support': int(np.count_nonzero(
                    (image_folds != held_fold) & primary_eligible & ~evc_eligible)),
                'primary_evaluation_images_missing_EVC_support': int(np.count_nonzero(
                    (image_folds == held_fold) & primary_eligible & ~evc_eligible)),
                'image_support': str(fold_dir / 'training_image_support.jsonl'),
                'repeat_assignments': str(fold_dir / 'repeat_assignments.npz'), 'rois': {}}
            report['folds'].append(fold_report)
            with (fold_dir / 'training_image_support.jsonl').open('w') as stream:
                for index in np.flatnonzero(image_folds != held_fold):
                    stream.write(json.dumps({'image_id': images[index]['image_id'], 'frozen_image_index': int(index),
                        'primary_valid_repeats': int(primary_counts[index]), 'EVC_valid_repeats': int(evc_counts[index]),
                        'primary_eligible': bool(primary_eligible[index]), 'common_context_image': bool(common[index]),
                        'context_exclusion_reason': None if common[index] else (
                            'primary_ineligible' if not primary_eligible[index] else 'EVC_missing_minimum_two')}) + '\n')
            np.savez(fold_dir / 'repeat_assignments.npz', common_frozen_image_indices=selected_indices,
                     columns=np.asarray(['session_index', 'session_row_index', 'common_image_index',
                                         'run', 'session_beta_axis_index', 'alternating_group']),
                     session_names=np.asarray(SESSIONS), primary=plans[0][0], EVC=plans[1][0])
            for roi, name in enumerate(ROI_NAMES):
                records, counts = plans[0 if roi < 3 else 1]
                columns = np.flatnonzero(support[:, roi])
                eta, nc, reliability = roi_context(sessions, means, scales, columns, records, len(selected_indices))
                np.savez(fold_dir / f'{name}_context.npz', voxel_indices_c=voxel_indices[columns],
                         eta=eta, provider_formula_NC_percent=nc, signed_repeat_correlation=reliability)
                image_sessions = np.unique(records[:, [2, 0]], axis=0)
                spanning = int(np.count_nonzero(np.bincount(
                    image_sessions[:, 0], minlength=len(selected_indices)) > 1))
                fold_report['rois'][name] = {
                    'retained_voxels': len(columns), 'common_training_images': len(selected_indices),
                    'repeat_assignment_group': 'primary' if roi < 3 else 'EVC',
                    'valid_training_presentations': len(records),
                    'repeat_count_histogram': dict(sorted(Counter(int(value) for value in counts).items())),
                    'h_mean_inverse_repeat_count': float(np.mean(1. / counts)) if len(counts) else None,
                    'presentations_by_session': {session: int(np.count_nonzero(records[:, 0] == index))
                                                 for index, session in enumerate(SESSIONS)},
                    'images_with_repeats_spanning_sessions': spanning,
                    'independent_repeat_errors_established': False,
                    'repeat_reliability': summary(reliability), 'provider_formula_NC_percent': summary(nc),
                    'zero_residual_noise_voxels': int(np.count_nonzero(eta == 0)),
                    'eta_at_least_one_voxels': int(np.count_nonzero(eta >= 1)),
                    'per_voxel_values': str(fold_dir / f'{name}_context.npz'),
                    'support_evaluable': bool(estimable and len(columns) >= 3 and len(selected_indices) >= 2),
                }
                save()
            fold_report['complete'] = True
            save()
        report['complete_training_context_components'] = len(report['folds']) == 5 and all(
            fold['complete'] for fold in report['folds'])
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    finally:
        signal.alarm(0)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report['complete_training_context_components'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
