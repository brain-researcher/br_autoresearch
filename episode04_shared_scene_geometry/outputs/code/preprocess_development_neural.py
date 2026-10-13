#!/usr/bin/env python3
"""Adopted foldwise neural preprocessing on complete development handoffs only."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import numpy as np

from inspect_array_metadata import PARTICIPANTS, read_json
from prepare_development_neural_participant import SESSIONS, artifact_presence, require_complete_session
from prepare_development_neural_pilot import EPISODE, ROI_NAMES, SCRATCH_LIMIT, SPATIAL_SHAPE


CHUNK_VOXELS = 256
FOLDS = range(5)
STRATA = ('laion', 'things', 'thingsplus')


@dataclass
class Session:
    name: str
    betas: np.ndarray
    image_indices: np.ndarray
    folds: np.ndarray
    rows: list[dict]


def training_statistics(sessions, held_fold, chunk_voxels=CHUNK_VOXELS, checkpoint=None):
    """Only training rows enter finite support, mean or population scale."""
    n_voxels = sessions[0].betas.shape[1]
    means = np.full((len(sessions), n_voxels), np.nan, dtype=np.float64)
    scales = np.full_like(means, np.nan)
    finite_all = np.ones(n_voxels, dtype=bool)
    positive_scale_all = np.ones(n_voxels, dtype=bool)
    counts = np.array([np.count_nonzero(session.folds != held_fold) for session in sessions], dtype=np.int64)
    if np.any(counts < 2):
        # The session-count condition is structural, not observed nonfiniteness
        # or zero variance. All masks are unassessed/false in this case; callers
        # must retain statistics_estimable=False rather than count nonfinite data.
        unassessed = np.zeros(n_voxels, dtype=bool)
        return means, scales, unassessed, unassessed.copy(), unassessed.copy(), counts
    for index, session in enumerate(sessions):
        training_rows = np.flatnonzero(session.folds != held_fold)
        for start in range(0, n_voxels, chunk_voxels):
            stop = min(start + chunk_voxels, n_voxels)
            values = np.asarray(session.betas[np.ix_(training_rows, np.arange(start, stop))], dtype=np.float64)
            finite = np.isfinite(values).all(axis=0)
            finite_all[start:stop] &= finite
            local_mean = np.full(stop - start, np.nan, dtype=np.float64)
            local_scale = np.full_like(local_mean, np.nan)
            with np.errstate(over='ignore', invalid='ignore'):
                local_mean[finite] = values[:, finite].mean(axis=0, dtype=np.float64)
                local_scale[finite] = values[:, finite].std(axis=0, ddof=0, dtype=np.float64)
            good_scale = finite & np.isfinite(local_mean) & np.isfinite(local_scale) & (local_scale > 0)
            positive_scale_all[start:stop] &= good_scale
            means[index, start:stop] = local_mean
            scales[index, start:stop] = local_scale
        if checkpoint is not None:
            checkpoint(index + 1)
    keep = finite_all & positive_scale_all
    return means, scales, keep, finite_all, positive_scale_all, counts


def retained_roi_support(membership, keep):
    return membership & keep[:, None]


def presentation_validity(sessions, means, scales, support, chunk_voxels=CHUNK_VOXELS):
    """Primary validity spans all three ROIs; EVC never alters it."""
    primary_adequate = bool(np.all(support[:, :3].sum(axis=0) >= 3))
    evc_adequate = bool(support[:, 3].sum() >= 3)
    groups = (np.flatnonzero(support[:, :3].any(axis=1)), np.flatnonzero(support[:, 3]))
    primary, evc = [], []
    for session_index, session in enumerate(sessions):
        validity = []
        for adequate, columns in zip((primary_adequate, evc_adequate), groups):
            valid = np.full(len(session.image_indices), adequate, dtype=bool)
            if adequate:
                for start in range(0, len(columns), chunk_voxels):
                    selected = columns[start:start + chunk_voxels]
                    values = np.asarray(session.betas[:, selected], dtype=np.float64)
                    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                        values = (values - means[session_index, selected]) / scales[session_index, selected]
                    valid &= np.isfinite(values).all(axis=1)
            validity.append(valid)
        primary.append(validity[0])
        evc.append(validity[1])
    return primary, evc


def repeat_counts(sessions, valid, image_count):
    counts = np.zeros(image_count, dtype=np.int64)
    for session, flags in zip(sessions, valid):
        counts += np.bincount(session.image_indices[flags], minlength=image_count)
    return counts


def image_eligibility(primary_counts, evc_counts):
    primary = primary_counts >= 2
    return primary, primary & (primary_counts >= 4), primary & (evc_counts >= 2)


def mean_pattern_block(sessions, means, scales, columns, valid, counts, eligible):
    """Each valid presentation has weight1, including unequal3/4-repeat images."""
    sums = np.zeros((len(counts), len(columns)), dtype=np.float64)
    for session_index, (session, flags) in enumerate(zip(sessions, valid)):
        selected_rows = np.flatnonzero(flags)
        if not len(selected_rows):
            continue
        values = np.asarray(session.betas[np.ix_(selected_rows, columns)], dtype=np.float64)
        values = (values - means[session_index, columns]) / scales[session_index, columns]
        if not np.isfinite(values).all():
            raise ValueError('Presentation validity and selected standardized values disagree')
        np.add.at(sums, session.image_indices[selected_rows], values)
    result = np.full(sums.shape, np.nan, dtype=np.float64)
    result[eligible] = sums[eligible] / counts[eligible, None]
    if not np.isfinite(result[eligible]).all():
        raise ValueError('Finite selected repetitions produced a nonfinite mean')
    return result


def completed_session_records(participant_report, participant):
    rows = participant_report.get('sessions', [])
    if (participant_report.get('participant') != participant
            or participant_report.get('complete_participant_handoff') is not True
            or participant_report.get('errors') != [] or len(rows) != 30
            or {row.get('session') for row in rows} != set(SESSIONS)
            or not all(row.get('complete') is True for row in rows)):
        raise ValueError('All30 complete session handoffs for this exact participant are required')
    return sorted(rows, key=lambda row: row['session'])


def read_json_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def load_completed_inputs(participant, participant_report_path, metadata_dir):
    # Authenticate every completion/role join before any beta computation.
    participant_report = read_json(participant_report_path)
    completed = completed_session_records(participant_report, participant)
    summary = read_json(metadata_dir / 'source_metadata_report.json')
    if summary.get('metadata_fold_freeze_complete') is not True:
        raise ValueError('Frozen development metadata is incomplete')
    images = [row for row in read_json_lines(metadata_dir / 'development_images.jsonl')
              if row['participant'] == participant]
    images.sort(key=lambda row: row['image_id'])
    if (len(images) != 4712 or len({row['image_id'] for row in images}) != 4712
            or any(type(row['fold']) is not int or row['fold'] not in FOLDS
                   or row['source_stratum'] not in STRATA for row in images)):
        raise ValueError('Exact participant image/fold/source metadata required')
    image_lookup = {row['image_id']: index for index, row in enumerate(images)}
    expected_trials = {session: [] for session in SESSIONS}
    with (metadata_dir / 'development_trials.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['participant'] == participant:
                expected_trials[row['session']].append(row)
    frozen_trials = {row['session']: row for row in summary['trial_files'] if row['participant'] == participant}
    occurrences = Counter()
    selected = []
    for item in completed:
        session = item['session']
        report = read_json(Path(item['report']))
        require_complete_session(report, participant, session, Path(summary['source_root']), frozen_trials[session])
        artifact_presence(report)
        payload = Path(report['payload_dir'])
        rows = read_json_lines(payload / 'development_trials.jsonl')
        expected = sorted(expected_trials[session], key=lambda row: row['session_beta_axis_index'])
        if rows != expected:
            raise ValueError('Candidate trial rows differ from the exact frozen development export')
        for row in rows:
            image = images[image_lookup[row['image_id']]]
            if any(row[key] != image[key] for key in ('participant', 'fold', 'component_id', 'source_stratum')):
                raise ValueError('Candidate trial/image/fold identity mismatch')
            occurrences[row['image_id']] += 1
        selected.append((session, payload, rows, report))
    if any(occurrences[row['image_id']] != row['observed_trial_metadata_rows'] for row in images):
        raise ValueError('Complete session handoffs do not reproduce frozen image-repeat counts')

    sessions, reference_indices, reference_membership = [], None, None
    for name, payload, rows, report in selected:
        indices = np.load(payload / 'voxel_indices_c.npy', allow_pickle=False)
        membership = np.load(payload / 'roi_membership.npy', allow_pickle=False)
        voxel_count = report['union_voxel_count']
        if (indices.shape != (voxel_count,) or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= np.prod(SPATIAL_SHAPE))
                or not np.all(np.diff(indices) > 0)
                or membership.shape != (voxel_count, 4) or membership.dtype != np.dtype(bool)
                or not membership.any(axis=1).all()
                or {name: int(membership[:, roi].sum()) for roi, name in enumerate(ROI_NAMES)} != report['roi_voxel_counts']):
            raise ValueError('Candidate voxel identities/registered ROI memberships disagree')
        if reference_indices is None:
            reference_indices, reference_membership = indices, membership
        elif not (np.array_equal(indices, reference_indices) and np.array_equal(membership, reference_membership)):
            raise ValueError('Exact voxel identities or ROI memberships differ across sessions')
        beta = np.load(payload / 'development_betas.npy', mmap_mode='r', allow_pickle=False)
        if beta.shape != (len(rows), voxel_count) or beta.dtype != np.dtype('float32'):
            raise ValueError('Candidate beta shape/dtype differs from the original-value handoff')
        sessions.append(Session(name, beta,
            np.array([image_lookup[row['image_id']] for row in rows], dtype=np.int64),
            np.array([row['fold'] for row in rows], dtype=np.int8), rows))
    return images, sessions, reference_indices, reference_membership


def write_mean_patterns(path, sessions, means, scales, columns, valid, counts, eligible):
    shape = (len(counts), len(columns))
    partial = path.with_suffix('.partial.npy')
    if not len(columns):
        np.save(partial, np.empty(shape, dtype=np.float64), allow_pickle=False)
    else:
        output = np.lib.format.open_memmap(partial, mode='w+', dtype='float64', shape=shape)
        for start in range(0, len(columns), CHUNK_VOXELS):
            stop = min(start + CHUNK_VOXELS, len(columns))
            output[:, start:stop] = mean_pattern_block(
                sessions, means, scales, columns[start:stop], valid, counts, eligible)
        output.flush()
        del output
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', choices=PARTICIPANTS, required=True)
    parser.add_argument('--participant-handoff', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--payload-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=840)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    scratch = os.environ.get('SCRATCH')
    if (not scratch or not args.payload_dir.resolve().is_relative_to(
            (Path(scratch) / 'br_autoresearch' / EPISODE).resolve())
            or args.payload_dir.exists() or args.output_dir.exists()
            or not 0 < args.max_seconds <= 840):
        raise ValueError('Fresh episode scratch/report paths and positive deadline<=840s required')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {'analysis_id': 'neural_preprocessing', 'participant': args.participant,
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'complete_preprocessing': False,
        'participant_handoff': str(args.participant_handoff), 'payload_dir': str(args.payload_dir),
        'phase': 'complete_handoff_identity', 'folds': [], 'errors': [],
        'original_inputs_preserved': True, 'source_mixed_arrays_opened': False,
        'embedding_RDM_reliability_NC_or_model_score_computed': False,
        'dtype': 'float64', 'voxel_chunk': CHUNK_VOXELS, 'scratch_limit_bytes': SCRATCH_LIMIT}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = args.output_dir / 'preprocessing.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output_dir / 'preprocessing.json')
        print(json.dumps({'phase': report['phase'], 'folds_recorded': len(report['folds']),
                          'elapsed_seconds': report['elapsed_seconds']}), flush=True)

    def deadline(*_):
        raise TimeoutError('Selected preprocessing internal time bound reached')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        images, sessions, voxel_indices, membership = load_completed_inputs(
            args.participant, args.participant_handoff, args.metadata_dir)
        # Bound all five folds before writing any numeric output. Overlapping
        # ROI voxels count twice because EVC may use different valid repeats.
        estimate = (5 * len(images) * int(membership.sum()) * 8
                    + 5 * len(sessions) * len(voxel_indices) * 8 * 2
                    + 5 * 8 * 1024**2)
        if estimate > SCRATCH_LIMIT:
            raise ValueError('Prospective complete preprocessing output exceeds16GiB')
        report['reserved_payload_upper_bytes'] = estimate
        report['image_count'] = len(images)
        report['session_count'] = len(sessions)
        report['union_voxel_count'] = len(voxel_indices)
        args.payload_dir.mkdir(parents=True)
        with (args.payload_dir / 'image_rows.jsonl').open('w') as output:
            for row in images:
                output.write(json.dumps(row, sort_keys=True) + '\n')
        np.save(args.payload_dir / 'voxel_indices_c.npy', voxel_indices, allow_pickle=False)
        np.save(args.payload_dir / 'source_roi_membership.npy', membership, allow_pickle=False)
        image_folds = np.array([row['fold'] for row in images])
        image_strata = np.array([row['source_stratum'] for row in images])
        for held_fold in FOLDS:
            fold_report = {'held_fold': held_fold, 'complete': False}
            report['folds'].append(fold_report)
            report['phase'] = 'training_session_statistics'
            save()
            def checkpoint(completed_sessions):
                fold_report['training_sessions_processed'] = completed_sessions
                if completed_sessions % 10 == 0:
                    save()
            means, scales, keep, finite, positive, counts = training_statistics(
                sessions, held_fold, checkpoint=checkpoint)
            support = retained_roi_support(membership, keep)
            statistics_estimable = bool(np.all(counts >= 2))
            primary_adequate = bool(np.all(support[:, :3].sum(axis=0) >= 3))
            evc_adequate = bool(support[:, 3].sum() >= 3)
            fold_report['training_statistics_estimable'] = statistics_estimable
            fold_report['training_presentation_counts_by_session'] = dict(zip(SESSIONS, counts.tolist()))
            fold_report['roi_support'] = {name: {
                'source_voxels': int(membership[:, roi].sum()),
                'retained_voxels': int(support[:, roi].sum()),
                'excluded_nonfinite_training': int(np.count_nonzero(membership[:, roi] & ~finite)) if statistics_estimable else 0,
                'excluded_scale_among_finite_training': int(np.count_nonzero(membership[:, roi] & finite & ~positive)) if statistics_estimable else 0,
                'excluded_unestimable_session_count': int(membership[:, roi].sum()) if np.any(counts < 2) else 0,
                'at_least_three_voxels': bool(support[:, roi].sum() >= 3)}
                for roi, name in enumerate(ROI_NAMES)}
            fold_report['training_sessions_with_fewer_than_two_presentations'] = [
                name for name, count in zip(SESSIONS, counts) if count < 2]
            report['phase'] = 'presentation_validity'
            save()
            primary_valid, evc_valid = presentation_validity(sessions, means, scales, support)
            primary_counts = repeat_counts(sessions, primary_valid, len(images))
            evc_counts = repeat_counts(sessions, evc_valid, len(images))
            primary_eligible, complete_four, evc_eligible = image_eligibility(primary_counts, evc_counts)
            fold_dir = args.payload_dir / f'fold-{held_fold}'
            fold_dir.mkdir()
            np.savez(fold_dir / 'training_parameters.npz', session_names=np.asarray(SESSIONS),
                     mean=means, scale=scales, training_counts=counts, retained_support=support,
                     statistics_estimable=np.asarray(statistics_estimable),
                     training_finite_all_sessions=finite, training_positive_scale_all_sessions=positive)
            np.savez(fold_dir / 'repeat_support.npz', primary_counts=primary_counts, evc_counts=evc_counts,
                     primary_eligible=primary_eligible, complete_four=complete_four,
                     evc_eligible_on_primary_images=evc_eligible)
            with (fold_dir / 'presentation_validity.jsonl').open('w') as output:
                for session, primary_flags, evc_flags in zip(sessions, primary_valid, evc_valid):
                    for row, p_valid, e_valid in zip(session.rows, primary_flags, evc_flags):
                        output.write(json.dumps({'session': session.name,
                            'session_beta_axis_index': row['session_beta_axis_index'],
                            'image_id': row['image_id'], 'primary_valid': bool(p_valid),
                            'evc_valid': bool(e_valid),
                            'primary_invalid_reason': None if p_valid else (
                                'insufficient_primary_voxel_support' if not primary_adequate
                                else 'nonfinite_standardized_primary_vector'),
                            'evc_invalid_reason': None if e_valid else (
                                'insufficient_EVC_voxel_support' if not evc_adequate
                                else 'nonfinite_standardized_EVC_vector')}) + '\n')
            with (fold_dir / 'image_validity.jsonl').open('w') as output:
                for index, row in enumerate(images):
                    output.write(json.dumps({'image_id': row['image_id'],
                        'fold_role': 'evaluation' if row['fold'] == held_fold else 'training',
                        'intended_repeats': row['intended_repeats'],
                        'observed_metadata_repeats': row['observed_trial_metadata_rows'],
                        'primary_valid_repeats': int(primary_counts[index]),
                        'evc_valid_repeats': int(evc_counts[index]),
                        'primary_eligible': bool(primary_eligible[index]),
                        'complete_four': bool(complete_four[index]),
                        'evc_eligible_on_primary_images': bool(evc_eligible[index]),
                        'primary_ineligible_reason': None if primary_eligible[index] else 'fewer_than_two_valid_presentations',
                        'evc_ineligible_reason': None if evc_eligible[index] else (
                            'primary_image_ineligible' if not primary_eligible[index] else 'fewer_than_two_EVC_presentations')}) + '\n')
            report['phase'] = 'equal_repeat_mean_patterns'
            save()
            for roi, name in enumerate(ROI_NAMES):
                columns = np.flatnonzero(support[:, roi])
                valid = primary_valid if roi < 3 else evc_valid
                image_counts = primary_counts if roi < 3 else evc_counts
                eligible = primary_eligible if roi < 3 else evc_eligible
                np.save(fold_dir / f'{name}_voxel_indices_c.npy', voxel_indices[columns], allow_pickle=False)
                write_mean_patterns(fold_dir / f'{name}_patterns.npy', sessions, means, scales,
                                    columns, valid, image_counts, eligible)
                fold_report.setdefault('pattern_ROIs_written', []).append(name)
                save()
            evaluation = image_folds == held_fold
            fold_report['evaluation_image_counts_by_stratum'] = {stratum: {
                'frozen': int(np.count_nonzero(evaluation & (image_strata == stratum))),
                'primary_min_two': int(np.count_nonzero(evaluation & (image_strata == stratum) & primary_eligible)),
                'complete_four': int(np.count_nonzero(evaluation & (image_strata == stratum) & complete_four))}
                for stratum in STRATA}
            fold_report['primary_cells_have_at_least_three_images'] = all(
                cell['primary_min_two'] >= 3 for cell in fold_report['evaluation_image_counts_by_stratum'].values())
            fold_report['complete_four_cells_have_at_least_three_images'] = all(
                cell['complete_four'] >= 3 for cell in fold_report['evaluation_image_counts_by_stratum'].values())
            fold_report['primary_evaluation_images_missing_EVC_support'] = int(
                np.count_nonzero(evaluation & primary_eligible & ~evc_eligible))
            fold_report['primary_support_evaluable'] = (
                all(fold_report['roi_support'][name]['at_least_three_voxels'] for name in ROI_NAMES[:3])
                and fold_report['primary_cells_have_at_least_three_images'])
            fold_report['payload_dir'] = str(fold_dir)
            fold_report['complete'] = True
            save()
        report['complete_preprocessing'] = all(fold['complete'] for fold in report['folds']) and len(report['folds']) == 5
        report['phase'] = 'completed_preprocessing_no_scores'
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
        report['phase_at_failure'] = report['phase']
        report['phase'] = 'incomplete'
    finally:
        signal.alarm(0)
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report['complete_preprocessing'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
