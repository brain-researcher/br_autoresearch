#!/usr/bin/env python3
"""Matched training-context RSA for one fixed participant, all 80 required rows.

Uses saved mean patterns and qualified development embeddings. No beta reader,
raw pixels, primary held-fold scores, model selection, fitting or null draws.
Existing native_rsa functions perform all geometry/rank/correlation arithmetic.
"""
import argparse
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
PARTICIPANTS = ('sub-01', 'sub-03', 'sub-05', 'sub-06', 'sub-07')
ROIS = ('laion-ventral', 'laion-lateral', 'laion-dorsal', 'retinotopic-EVC')
MODELS = {
    'CLIP': (1024, 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'),
    'DINOv2': (1024, 'vit_large_patch14_dinov2'),
    'PEcore': (1024, 'PE-Core-L14-336'),
    'SigLIP2': (1152, 'google/siglip2-so400m-patch14-384'),
}
EMBEDDINGS = Path('outputs/runs/source_readiness/20261003-development_embeddings01')
CONTEXT_REPORTS = {
    'sub-01': Path('outputs/runs/training_neural_context/20261005-sub01-context-46654794/training_context.json'),
    'sub-05': Path('outputs/runs/training_neural_context/20261003-sub05-folds01/training_context.json'),
    **{person: Path('outputs/runs/training_neural_context/20261005-remaining-context-46655460')
       / person / 'training_context.json' for person in ('sub-03', 'sub-06', 'sub-07')},
}
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def preprocessing_report(participant):
    if participant == 'sub-05':
        return Path('outputs/runs/neural_preprocessing/20261003-sub05-folds01/preprocessing.json')
    return Path('outputs/runs/neural_preprocessing/20261003-remaining-folds01') / participant / 'preprocessing.json'


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def matched_support(np, images, fold_dir, context_fold, held_fold):
    """Bind the existing context's exact images and repeats to saved mean inputs."""
    validity = json_lines(fold_dir / 'image_validity.jsonl')
    if (len(validity) != len(images)
            or any(actual['image_id'] != expected['image_id'] for actual, expected in zip(validity, images))):
        raise ValueError('Saved pattern image order and validity differ')
    expected_indices = np.array([
        index for index, (image, row) in enumerate(zip(images, validity))
        if image['fold'] != held_fold and row['primary_eligible'] is True
        and row['evc_eligible_on_primary_images'] is True
    ], dtype=np.int64)
    with np.load(Path(context_fold['repeat_assignments']), allow_pickle=False) as saved:
        indices = saved['common_frozen_image_indices']
        sessions = saved['session_names'].tolist()
        if (indices.dtype.kind not in 'iu' or indices.ndim != 1
                or not np.array_equal(indices, expected_indices)
                or sessions != [f'ses-{number:02d}' for number in range(1, 31)]
                or saved['columns'].tolist() != ['session_index', 'session_row_index', 'common_image_index',
                                                'run', 'session_beta_axis_index', 'alternating_group']):
            raise ValueError('Context training image/repeat-axis identity differs')
        assignments = {name: saved[name] for name in ('primary', 'EVC')}
    if len(indices) < 3 or context_fold['common_training_images'] != len(indices):
        raise ValueError('Required matched training image support is incomplete')
    selected_ids = {images[int(index)]['image_id'] for index in indices}
    training_support = json_lines(Path(context_fold['image_support']))
    expected_training = {index for index, row in enumerate(images) if row['fold'] != held_fold}
    if ({row['frozen_image_index'] for row in training_support} != expected_training
            or len(training_support) != len(expected_training)):
        raise ValueError('Context support table does not contain the exact training images')
    for row in training_support:
        index = row['frozen_image_index']
        original = validity[index]
        if (row['image_id'] != images[index]['image_id']
                or row['common_context_image'] is not (images[index]['image_id'] in selected_ids)
                or row['primary_valid_repeats'] != original['primary_valid_repeats']
                or row['EVC_valid_repeats'] != original['evc_valid_repeats']):
            raise ValueError('Context image identity or repeat counts differ from mean-pattern inputs')

    # Mean-pattern construction used these same saved presentation flags. Match
    # actual session/row/image/beta-axis identities, not just equal repeat counts.
    presentations = json_lines(fold_dir / 'presentation_validity.jsonl')
    positions = {}
    session_rows = {name: 0 for name in sessions}
    for row in presentations:
        session = row['session']
        if session not in session_rows:
            raise ValueError('Unknown session in saved presentation validity')
        index = session_rows[session]
        positions[(sessions.index(session), index)] = row
        session_rows[session] += 1
    for group, flag, count_key in (('primary', 'primary_valid', 'primary_valid_repeats'),
                                    ('EVC', 'evc_valid', 'evc_valid_repeats')):
        records = assignments[group]
        if records.ndim != 2 or records.shape[1] != 6 or records.dtype.kind not in 'iu':
            raise ValueError('Malformed context repeat assignment')
        expected = {key for key, row in positions.items()
                    if row[flag] is True and row['image_id'] in selected_ids}
        observed = set()
        for record in records:
            session_index, session_row, image_index, _run, beta_axis, _half = map(int, record)
            key = (session_index, session_row)
            if key in observed or key not in expected or not 0 <= image_index < len(indices):
                raise ValueError('Context selected-repeat identity differs from saved mean support')
            row = positions[key]
            if (row['image_id'] != images[int(indices[image_index])]['image_id']
                    or row['session_beta_axis_index'] != beta_axis):
                raise ValueError('Context beta-axis/image join differs from saved mean support')
            observed.add(key)
        counts = np.bincount(records[:, 2], minlength=len(indices))
        if observed != expected or not np.array_equal(
                counts, [validity[int(index)][count_key] for index in indices]):
            raise ValueError('Context omits or duplicates eligible saved presentations')
    return indices


def ranked_geometry(native, values, distance):
    matrix = native.rdm(values, distance=distance)
    upper = native.upper_triangle(matrix)
    del matrix
    ranks = native.average_ranks(upper)
    del upper
    return ranks


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', choices=PARTICIPANTS, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=1500)
    args = parser.parse_args()
    hostname = socket.gethostname()
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in hostname
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(key) != '1' for key in THREAD_ENV)):
        raise RuntimeError('Use one actual compute CPU and all six numerical thread limits=1')
    if Path.cwd().resolve() != EPISODE or not 0 < args.max_seconds <= 1500:
        raise ValueError('Use EP04 cwd and a positive whole-process cap<=1500s')
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(EPISODE / 'outputs/runs/training_context_rsa'):
        raise ValueError('Use a fresh EP04 training_context_rsa output directory')
    output.mkdir(parents=True)
    report = {
        'analysis_id': 'training_neural_context', 'mode': 'matched_training_context_RSA',
        'participant': args.participant, 'job_id': os.environ['SLURM_JOB_ID'], 'hostname': hostname,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'internal_seconds': args.max_seconds,
        'status': 'started', 'complete_participant_training_context_RSA': False,
        'all_participants_complete': False, 'primary_panel_evaluated': False,
        'candidate_selection_performed': False, 'held_out_performance': False,
        'raw_beta_arrays_read': False, 'raw_images_or_excluded_outcomes_opened': False,
        'preprocessing_or_noise_ceiling_refitted': False, 'null_draws': 0,
        'automatic_retry': False, 'automatic_resume': False,
        'scientific_interpretation': 'Descriptive matched training-image context; NC is not a strict Spearman ceiling',
        'preprocessing_report': str(preprocessing_report(args.participant)),
        'context_report': str(CONTEXT_REPORTS[args.participant]),
        'embedding_handoff': str(EMBEDDINGS / 'embedding_handoff.json'),
        'folds': [], 'rows': [], 'errors': [],
    }

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('Matched training-context RSA whole-process deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        pending = output / 'training_context_rsa.inprogress.json'
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        pending.replace(output / 'training_context_rsa.json')

    def progress(stage, **details):
        check()
        report['progress'] = {'stage': stage, 'elapsed_seconds': time.monotonic() - started, **details}
        save()
        print(json.dumps(report['progress'], allow_nan=False), flush=True)
        check()

    def interrupted(signum, _frame):
        raise TimeoutError(f'Matched training-context RSA interrupted by signal {signum}')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(max(1, math.ceil(args.max_seconds - (time.monotonic() - started))))
    try:
        progress('before_qualified_helper_import')
        import numpy as np
        import native_rsa as native
        progress('before_completed_metadata')
        prepared = json.loads(preprocessing_report(args.participant).read_text())
        context = json.loads(CONTEXT_REPORTS[args.participant].read_text())
        if (prepared.get('participant') != args.participant or prepared.get('complete_preprocessing') is not True
                or prepared.get('errors') != [] or context.get('participant') != args.participant
                or context.get('complete_training_context_components') is not True or context.get('errors') != []
                or Path(context['preprocessing_report']).resolve() != preprocessing_report(args.participant).resolve()):
            raise ValueError('Completed matching preprocessing/context required')
        for item in (prepared, context):
            if (len(item.get('folds', [])) != 5
                    or {fold['held_fold'] for fold in item['folds']} != set(range(5))
                    or not all(fold.get('complete') is True for fold in item['folds'])):
                raise ValueError('All five frozen folds must be completed')
        payload = Path(prepared['payload_dir'])
        images = json_lines(payload / 'image_rows.jsonl')
        if (len(images) != 4712 or len({row['image_id'] for row in images}) != 4712
                or any(row['participant'] != args.participant or type(row['fold']) is not int
                       or row['fold'] not in range(5) for row in images)):
            raise ValueError('Fixed participant image identities differ')
        embedding_report = json.loads((EMBEDDINGS / 'embedding_handoff.json').read_text())
        if (embedding_report.get('complete_development_embedding_handoff') is not True
                or embedding_report.get('development_rows') != 23560
                or embedding_report.get('errors') != []
                or embedding_report.get('row_identity_file') != 'development_rows.jsonl'):
            raise ValueError('Qualified complete development embedding handoff required')
        embedding_rows = json_lines(EMBEDDINGS / 'development_rows.jsonl')
        embedding_index = {row['image_id']: row for row in embedding_rows}
        if (len(embedding_rows) != 23560 or len(embedding_index) != 23560
                or any(row['output_row'] != index for index, row in enumerate(embedding_rows))):
            raise ValueError('Embedding image/output-row identity differs')
        for row in images:
            feature = embedding_index.get(row['image_id'])
            if feature is None or any(feature[key] != row[key]
                                      for key in ('participant', 'fold', 'source_stratum', 'component_id')):
                raise ValueError('Embedding and pattern image/fold/component identities differ')
        declarations = {item['model_label']: item for item in embedding_report['models']}
        if set(declarations) != set(MODELS) or len(embedding_report['models']) != 4:
            raise ValueError('Exactly four fixed released models required')
        features = {}
        for model, (dimension, model_id) in MODELS.items():
            item = declarations[model]
            if (item['shape'] != [23560, dimension] or item['dtype'] != 'float16'
                    or item['completed_rows'] != 23560 or item['output'] != f'{model}.npy'
                    or item['source_declarations']['model_id'] != model_id):
                raise ValueError('Frozen embedding/model declaration differs')
            features[model] = np.load(EMBEDDINGS / item['output'], mmap_mode='r', allow_pickle=False)
            if features[model].shape != (23560, dimension) or features[model].dtype != np.dtype('float16'):
                raise ValueError('Qualified embedding header differs')
        report['models'] = [dict(model_label=model, **declarations[model]['source_declarations']) for model in MODELS]

        for held_fold in range(5):
            progress('before_matched_training_support', held_fold=held_fold)
            fold = next(item for item in context['folds'] if item['held_fold'] == held_fold)
            fold_dir = payload / f'fold-{held_fold}'
            if set(fold['rois']) != set(ROIS) or not all(
                    row.get('support_evaluable') is True and row.get('complete', True) is True
                    for row in fold['rois'].values()):
                raise ValueError('All four completed/evaluable context ROIs required')
            indices = matched_support(np, images, fold_dir, fold, held_fold)
            feature_indices = np.array([embedding_index[images[int(index)]['image_id']]['output_row']
                                        for index in indices], dtype=np.int64)
            identity_path = output / f'fold-{held_fold}-training_images.jsonl'
            with identity_path.open('x') as stream:
                for source_index, feature_index in zip(indices, feature_indices):
                    stream.write(json.dumps({'image_id': images[int(source_index)]['image_id'],
                                             'pattern_row': int(source_index),
                                             'embedding_row': int(feature_index)}) + '\n')
            fold_result = {'held_fold': held_fold, 'training_images': len(indices),
                           'training_image_id_file': str(identity_path),
                           'matched_image_and_repeat_support': True, 'complete': False, 'rows_recorded': 0}
            report['folds'].append(fold_result)
            ranks, model_errors = {}, {}
            for model in MODELS:
                progress('before_model_geometry_and_rank', held_fold=held_fold, model=model)
                try:
                    ranks[model] = ranked_geometry(native, features[model][feature_indices, :], 'cosine')
                except native.NotEvaluable as exc:
                    model_errors[model] = str(exc)
                progress('after_model_geometry_and_rank', held_fold=held_fold, model=model)

            for roi in ROIS:
                progress('before_neural_geometry_and_rank', held_fold=held_fold, roi=roi)
                roi_context = fold['rois'][roi]
                voxel_indices = np.load(fold_dir / f'{roi}_voxel_indices_c.npy', allow_pickle=False)
                with np.load(Path(roi_context['per_voxel_values']), allow_pickle=False) as stored:
                    if not np.array_equal(stored['voxel_indices_c'], voxel_indices):
                        raise ValueError('Context and mean-pattern voxel identities differ')
                if len(voxel_indices) != roi_context['retained_voxels']:
                    raise ValueError('Context retained-voxel count differs')
                patterns = np.load(fold_dir / f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
                if patterns.shape != (len(images), len(voxel_indices)) or patterns.dtype != np.dtype('float64'):
                    raise ValueError('Saved mean-pattern header differs')
                neural_error = None
                neural_ranks = None
                try:
                    neural_ranks = ranked_geometry(native, patterns[indices, :], 'correlation')
                except native.NotEvaluable as exc:
                    neural_error = str(exc)
                del patterns
                progress('after_neural_geometry_and_rank', held_fold=held_fold, roi=roi)
                for model in MODELS:
                    check()
                    row = {'participant': args.participant, 'held_fold': held_fold, 'roi': roi,
                           'model_label': model, 'model_id': MODELS[model][1],
                           'training_images': len(indices), 'retained_voxels': len(voxel_indices),
                           'training_image_id_file': str(identity_path),
                           'context_per_voxel_artifact': roi_context['per_voxel_values'],
                           'valid_training_presentations': roi_context['valid_training_presentations'],
                           'repeat_count_histogram': roi_context['repeat_count_histogram'],
                           'repeat_reliability': roi_context['repeat_reliability'],
                           'provider_formula_NC_percent': roi_context['provider_formula_NC_percent'],
                           'independent_repeat_errors_established': False,
                           'rsa_spearman_r': None, 'rsa_fisher_z': None, 'status': 'not_evaluable'}
                    reason = neural_error or model_errors.get(model)
                    if reason is None:
                        try:
                            row['rsa_spearman_r'] = native.pearson(ranks[model], neural_ranks)
                            row['rsa_fisher_z'] = native.fisher_z(row['rsa_spearman_r'])
                            row['status'] = 'evaluable'
                        except native.NotEvaluable as exc:
                            reason = str(exc)
                    if reason is not None:
                        row['not_evaluable_reason'] = reason
                    report['rows'].append(row)
                    fold_result['rows_recorded'] += 1
                del neural_ranks
                progress('roi_rows_recorded', held_fold=held_fold, roi=roi)
            del ranks
            fold_result['complete'] = fold_result['rows_recorded'] == 16
            progress('fold_rows_recorded', held_fold=held_fold)
        check()
        report['all_required_rows_recorded'] = len(report['rows']) == 80 and all(fold['complete'] for fold in report['folds'])
        report['complete_participant_training_context_RSA'] = report['all_required_rows_recorded'] and all(
            row['status'] == 'evaluable' for row in report['rows'])
        report['status'] = ('participant_training_context_RSA_completed' if
                            report['complete_participant_training_context_RSA'] else
                            'participant_training_context_RSA_not_evaluable')
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['complete_participant_training_context_RSA'] = False
        report['status'] = 'participant_training_context_RSA_incomplete'
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() - started >= args.max_seconds:
            report['complete_participant_training_context_RSA'] = False
            report['status'] = 'participant_training_context_RSA_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_participant_training_context_RSA'] and time.monotonic() - started >= args.max_seconds:
            report['complete_participant_training_context_RSA'] = False
            report['status'] = 'participant_training_context_RSA_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_participant_training_context_RSA'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
