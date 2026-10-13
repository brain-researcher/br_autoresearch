#!/usr/bin/env python3
"""Complete EP04 observed evaluation-fold native and low-level partial RSA.

Uses existing development mean patterns, frozen embeddings and the completed
RGB block-moment handoff. No raw beta/image reader, neural refit, null draw,
candidate screen, pruning, winner, configuration lock or audit access.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_training_context_rsa as saved


EPISODE = saved.EPISODE
PARTICIPANTS = saved.PARTICIPANTS
ROIS = saved.ROIS
MODELS = saved.MODELS
METADATA = EPISODE / 'outputs/runs/source_readiness/20261003-metadata_folds04'
STRATA = ('laion', 'things', 'thingsplus')
VLM_LABELS = ('CLIP', 'PEcore', 'SigLIP2')
MAX_SECONDS = 2400


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def complete_features(np, low_module, low_dir, check):
    """Complete metadata/finite coverage before the first observed score."""
    summary = read_json(METADATA / 'source_metadata_report.json')
    images = saved.json_lines(METADATA / 'development_images.jsonl')
    expected = {row['image_id']: row for row in images}
    require(summary.get('metadata_fold_freeze_complete') is True
            and summary.get('development_images') == 23560
            and len(images) == len(expected) == 23560
            and Counter(row['participant'] for row in images)
            == Counter({person: 4712 for person in PARTICIPANTS})
            and all(type(row['fold']) is int and row['fold'] in range(5)
                    and row['source_stratum'] in STRATA for row in images),
            'Complete frozen development identities/folds required')
    handoff = read_json(saved.EMBEDDINGS / 'embedding_handoff.json')
    rows = saved.json_lines(saved.EMBEDDINGS / 'development_rows.jsonl')
    index = {row['image_id']: row for row in rows}
    require(handoff.get('complete_development_embedding_handoff') is True
            and handoff.get('development_rows') == 23560 and handoff.get('errors') == []
            and handoff.get('row_identity_file') == 'development_rows.jsonl'
            and len(rows) == len(index) == 23560 and set(index) == set(expected)
            and all(row['output_row'] == ordinal for ordinal, row in enumerate(rows)),
            'Complete frozen embedding handoff required')
    for row in rows:
        require(all(row[key] == expected[row['image_id']][key]
                    for key in ('participant', 'fold', 'source_stratum', 'component_id')),
                'Embedding image/fold/source/component identity differs')
    declarations = {item['model_label']: item for item in handoff['models']}
    require(len(handoff['models']) == 4 and set(declarations) == set(MODELS),
            'Exactly four frozen released models required')
    features = {}
    for model, (dimension, model_id) in MODELS.items():
        item = declarations[model]
        require(item['shape'] == [23560, dimension] and item['dtype'] == 'float16'
                and item['completed_rows'] == 23560 and item['output'] == f'{model}.npy'
                and item['source_declarations']['model_id'] == model_id,
                'Released model binding differs')
        array = np.load(saved.EMBEDDINGS / item['output'], mmap_mode='r', allow_pickle=False)
        require(array.shape == (23560, dimension) and array.dtype == np.dtype('float16'),
                'Embedding shape/dtype differs')
        for start in range(0, 23560, 256):
            check()
            block = np.asarray(array[start:start+256], dtype=np.float64)
            require(np.isfinite(block).all() and np.all(np.linalg.norm(block, axis=1) > 0),
                    'Every frozen embedding row must be finite and nonzero; no feature exclusions')
        features[model] = array
    low = read_json(low_dir / 'low_level_preparation_report.json')
    identifiers = read_json(low_dir / 'image_ids.json')
    raw_source = (Path(os.environ['OAK']) / 'users/zijiao/br_autoresearch_data/steward_acquisition'
                  / 'laion_fmri_raw_stimuli_dua_20260922/payload')
    require(low.get('control_id') == low_module.CONTROL_ID and low.get('mode') == 'full'
            and low.get('status') == 'complete' and low.get('complete_development_coverage') is True
            and low.get('selected') == low.get('completed') == 23560
            and low.get('stored_dtype') == '<f8' and low.get('feature_dimension') == 3150
            and low.get('excluded_regular_shared_or_ood_images_read') is False
            and low.get('neural_values_read') is False and low.get('candidate_scores_computed') is False
            and Path(low['metadata_dir']).resolve() == METADATA.resolve()
            and Path(low['raw_h5']).resolve() == (raw_source / 'task-images_stimuli.h5').resolve()
            and Path(low['raw_metadata']).resolve() == (raw_source / 'task-images_metadata.csv').resolve()
            and identifiers == sorted(expected),
            'Complete finite 23560-image approved low-level handoff required before scoring')
    low_features = np.load(low_dir / 'low_level_features.npy', mmap_mode='r', allow_pickle=False)
    require(low_features.shape == (23560, 3150) and low_features.dtype == np.dtype('<f8'),
            'Low-level feature shape/dtype differs')
    for start in range(0, 23560, 256):
        check()
        require(np.isfinite(low_features[start:start+256]).all(),
                'Nonfinite low-level feature row; no image may be dropped')
    return expected, index, declarations, features, {key: i for i, key in enumerate(identifiers)}, low_features


def participant_inputs(np, person, expected, check):
    prepared = read_json(saved.preprocessing_report(person))
    context = read_json(saved.CONTEXT_REPORTS[person])
    require(prepared.get('participant') == context.get('participant') == person
            and prepared.get('complete_preprocessing') is True and prepared.get('errors') == []
            and context.get('complete_training_context_components') is True and context.get('errors') == []
            and Path(context['preprocessing_report']).resolve() == saved.preprocessing_report(person).resolve(),
            'Completed participant preprocessing and training context required')
    for record in (prepared, context):
        require(len(record.get('folds', [])) == 5
                and {item['held_fold'] for item in record['folds']} == set(range(5))
                and all(item.get('complete') is True for item in record['folds']),
                'Every frozen preprocessing/context fold must be complete')
    payload = Path(prepared['payload_dir']).resolve()
    require(payload.is_relative_to(Path(os.environ['SCRATCH']) / 'br_autoresearch' / EPISODE.name),
            'Only the declared EP04 preprocessing payload is eligible')
    images = saved.json_lines(payload / 'image_rows.jsonl')
    require(images == sorted((row for row in expected.values() if row['participant'] == person),
                             key=lambda row: row['image_id']),
            'Saved mean-pattern rows differ from exact frozen image metadata')
    folds = []
    for held in range(5):
        check()
        original = next(item for item in prepared['folds'] if item['held_fold'] == held)
        training = next(item for item in context['folds'] if item['held_fold'] == held)
        directory = payload / f'fold-{held}'
        require(Path(original['payload_dir']).resolve() == directory
                and original['training_statistics_estimable'] is True
                and set(original['roi_support']) == set(ROIS) and set(training['rois']) == set(ROIS),
                'Saved fold/support identity differs')
        validity = saved.json_lines(directory / 'image_validity.jsonl')
        require(len(validity) == 4712, 'Full saved image validity required')
        for image, row in zip(images, validity):
            require(row['image_id'] == image['image_id']
                    and row['fold_role'] == ('evaluation' if image['fold'] == held else 'training')
                    and type(row['primary_valid_repeats']) is int and row['primary_valid_repeats'] >= 0
                    and type(row['evc_valid_repeats']) is int and row['evc_valid_repeats'] >= 0
                    and row['primary_eligible'] is (row['primary_valid_repeats'] >= 2)
                    and row['evc_eligible_on_primary_images'] is
                    (row['primary_eligible'] and row['evc_valid_repeats'] >= 2),
                    'Saved image role or fixed repeat eligibility differs')
        indices = np.array([i for i, (image, row) in enumerate(zip(images, validity))
                            if image['fold'] == held and row['primary_eligible']], dtype=np.int64)
        evc_missing = [images[int(i)]['image_id'] for i in indices
                       if not validity[int(i)]['evc_eligible_on_primary_images']]
        source_counts = {source: sum(images[int(i)]['source_stratum'] == source for i in indices)
                         for source in STRATA}
        require(all(source_counts[source] == original['evaluation_image_counts_by_stratum'][source]['primary_min_two']
                    for source in STRATA)
                and len(evc_missing) == original['primary_evaluation_images_missing_EVC_support'],
                'Saved evaluation support counts differ')
        primary_reason = None if len(indices) >= 3 and min(source_counts.values()) >= 3 else 'Insufficient fixed evaluation image/source support'
        folds.append({'held_fold': held, 'directory': directory, 'indices': indices,
                      'validity': validity, 'source_counts': source_counts,
                      'evc_missing': evc_missing, 'primary_reason': primary_reason,
                      'original': original, 'training': training})
    return images, folds


def effect_table(np, native, rows, prefix):
    keyed = {(row['model_label'], row['participant'], row['held_fold'], row['roi']): row for row in rows}
    if any(keyed[(model, person, fold, roi)][f'{prefix}_status'] != 'evaluable'
           for model in MODELS for person in PARTICIPANTS for fold in range(5) for roi in ROIS[:3]):
        return None
    def values(model):
        return np.array([[[keyed[(model, person, fold, roi)][f'{prefix}_fisher_z']
                           for roi in ROIS[:3]] for fold in range(5)] for person in PARTICIPANTS])
    calculated = native.panel_effects(np.array([values(model) for model in VLM_LABELS]), values('DINOv2'))
    return {'VLM_order': list(VLM_LABELS), 'participant_order': list(PARTICIPANTS),
            'primary_ROI_order': list(ROIS[:3]), 'fold_order': list(range(5)),
            'candidate_screen_performed': False,
            **{key: value.tolist() for key, value in calculated.items()}}


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-level-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    hostname = socket.gethostname()
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in hostname
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(key) != '1' for key in saved.THREAD_ENV)):
        raise RuntimeError('Use one actual compute CPU and six numerical thread limits=1')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
            'Use EP04 cwd and an explicit positive whole-process cap<=2400s')
    output, low_dir = args.output_dir.resolve(), args.low_level_dir.resolve()
    require(not output.exists() and output.is_relative_to(EPISODE / 'outputs/runs/observed_native_panel')
            and low_dir.is_relative_to(EPISODE / 'outputs/runs/source_readiness'),
            'Fresh EP04 observed-panel output and episode-local low-level handoff required')
    output.mkdir(parents=True)
    report = {'analysis_id': 'fixed_panel', 'mode': 'complete_observed_native_and_low_level_partial_RSA',
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': hostname,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'first_panel_evaluation_utc': None,
              'status': 'started', 'complete_observed_panel': False, 'all_400_rows_recorded': False,
              'complete_primary_observed_panel': False, 'complete_EVC_observed_diagnostic': False,
              'EVC_partial_is_descriptive_only': True,
              'candidate_selection_performed': False, 'candidate_lock_authorized': False,
              'null_draws': 0, 'remaining_sensitivities_or_encoding_performed': False,
              'primary_support_pruned_for_EVC': False, 'raw_beta_arrays_read': False,
              'raw_images_or_excluded_outcomes_opened': False, 'preprocessing_or_NC_refitted': False,
              'evidence': 'Evaluation images are held out within each fold; the complete development panel remains selection-exposed',
              'reliability_context': 'Frozen training-only context; not re-estimated on evaluation images or a strict Spearman ceiling',
              'low_level_report': str(low_dir / 'low_level_preparation_report.json'),
              'internal_seconds': args.max_seconds, 'automatic_retry': False, 'automatic_resume': False,
              'folds': [], 'rows': [], 'errors': []}

    def check():
        if time.monotonic() - started >= args.max_seconds:
            raise TimeoutError('Observed native panel whole-process deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        partial = output / 'observed_native_panel.inprogress.json'
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        partial.replace(output / 'observed_native_panel.json')

    def progress(phase, **fields):
        check()
        report['phase'] = phase
        save()
        print(json.dumps({'phase': phase, 'rows_recorded': len(report['rows']),
                          'elapsed_seconds': report['elapsed_seconds'], **fields}), flush=True)
        check()

    def interrupted(*_):
        raise TimeoutError('Observed native panel interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.max_seconds-(time.monotonic()-started)))
        progress('complete_feature_coverage_before_scores')
        import numpy as np
        import native_rsa as native
        import low_level_pixels_proposal as low_module
        expected, embedding_index, declarations, features, low_index, low_features = complete_features(np, low_module, low_dir, check)
        inputs = {person: participant_inputs(np, person, expected, check) for person in PARTICIPANTS}
        report['full_feature_coverage_qualified_before_scores'] = True
        report['models'] = [dict(model_label=model, **declarations[model]['source_declarations']) for model in MODELS]
        report['preprocessing_reports'] = {person: str(saved.preprocessing_report(person)) for person in PARTICIPANTS}
        report['training_context_reports'] = {person: str(saved.CONTEXT_REPORTS[person]) for person in PARTICIPANTS}
        report['first_panel_evaluation_utc'] = datetime.now(timezone.utc).isoformat()
        progress('begin_complete_observed_panel')
        for person in PARTICIPANTS:
            images, folds = inputs[person]
            for fold in folds:
                held, indices, directory = fold['held_fold'], fold['indices'], fold['directory']
                feature_rows = np.array([embedding_index[images[int(i)]['image_id']]['output_row'] for i in indices])
                low_rows = np.array([low_index[images[int(i)]['image_id']] for i in indices])
                identity_path = output / f'{person}-fold-{held}-evaluation_images.jsonl'
                with identity_path.open('x') as stream:
                    for i, feature_row, low_row in zip(indices, feature_rows, low_rows):
                        image, validity = images[int(i)], fold['validity'][int(i)]
                        stream.write(json.dumps({'image_id': image['image_id'], 'pattern_row': int(i),
                            'embedding_row': int(feature_row), 'low_level_row': int(low_row),
                            'source_stratum': image['source_stratum'], 'component_id': image['component_id'],
                            'primary_valid_repeats': validity['primary_valid_repeats'],
                            'evc_valid_repeats': validity['evc_valid_repeats']}) + '\n')
                folded = {'participant': person, 'held_fold': held, 'evaluation_images': len(indices),
                          'image_id_file': str(identity_path), 'source_counts': fold['source_counts'],
                          'EVC_missing_image_ids': fold['evc_missing'], 'complete': False}
                report['folds'].append(folded)
                matrices, ranks, model_errors = {}, {}, {}
                low_matrix, low_error = None, fold['primary_reason']
                if low_error is None:
                    progress('low_level_evaluation_geometry', participant=person, held_fold=held)
                    low_matrix = low_module.low_level_rdm(low_features[low_rows, :])
                    low_upper = native.upper_triangle(low_matrix)
                    if np.max(low_upper) == np.min(low_upper):
                        low_error = 'Constant low-level RDM is not evaluable'
                    del low_upper
                for model in MODELS:
                    progress('native_evaluation_geometry', participant=person, held_fold=held, model=model)
                    try:
                        if fold['primary_reason']:
                            raise native.NotEvaluable(fold['primary_reason'])
                        matrices[model] = native.rdm(features[model][feature_rows, :], 'cosine')
                        ranks[model] = native.average_ranks(native.upper_triangle(matrices[model]))
                    except native.NotEvaluable as exc:
                        model_errors[model] = str(exc)
                for roi in ROIS:
                    progress('neural_evaluation_geometry', participant=person, held_fold=held, roi=roi)
                    trained = fold['training']['rois'][roi]
                    voxels = np.load(directory / f'{roi}_voxel_indices_c.npy', allow_pickle=False)
                    require(voxels.ndim == 1 and voxels.dtype.kind in 'iu'
                            and len(set(voxels.tolist())) == len(voxels)
                            and len(voxels) == fold['original']['roi_support'][roi]['retained_voxels']
                            and len(voxels) == trained['retained_voxels'], 'ROI voxel support differs')
                    with np.load(Path(trained['per_voxel_values']), allow_pickle=False) as stored:
                        require(np.array_equal(stored['voxel_indices_c'], voxels),
                                'Training context and mean-pattern voxel identities differ')
                    neural_matrix, neural_ranks = None, None
                    neural_error = fold['primary_reason']
                    if len(voxels) < 3:
                        neural_error = 'Fewer than three training-qualified ROI voxels'
                    if roi == ROIS[3] and fold['evc_missing']:
                        neural_error = 'EVC lacks repeats on required primary IDs; no intersection substitution'
                    if neural_error is None:
                        patterns = np.load(directory / f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
                        require(patterns.shape == (4712, len(voxels)) and patterns.dtype == np.dtype('float64'),
                                'Saved mean-pattern header differs')
                        try:
                            neural_matrix = native.rdm(patterns[indices, :], 'correlation')
                            neural_ranks = native.average_ranks(native.upper_triangle(neural_matrix))
                        except native.NotEvaluable as exc:
                            neural_error = str(exc)
                        del patterns
                    for model in MODELS:
                        check()
                        row = {'participant': person, 'held_fold': held, 'roi': roi,
                               'roi_role': 'primary' if roi in ROIS[:3] else 'diagnostic',
                               'model_label': model, 'model_id': MODELS[model][1],
                               'evaluation_images': len(indices), 'evaluation_image_id_file': str(identity_path),
                               'retained_voxels': len(voxels),
                               'training_context_common_images': fold['training']['common_training_images'],
                               'training_repeat_reliability': trained['repeat_reliability'],
                               'training_provider_formula_NC_percent': trained['provider_formula_NC_percent'],
                               'native_status': 'not_evaluable', 'native_spearman_r': None, 'native_fisher_z': None,
                               'partial_status': 'not_evaluable', 'partial_spearman_r': None, 'partial_fisher_z': None}
                        reason = neural_error or model_errors.get(model)
                        if reason is None:
                            try:
                                row['native_spearman_r'] = native.pearson(ranks[model], neural_ranks)
                                row['native_fisher_z'] = native.fisher_z(row['native_spearman_r'])
                                row['native_status'] = 'evaluable'
                            except native.NotEvaluable as exc:
                                row['native_reason'] = str(exc)
                        else:
                            row['native_reason'] = reason
                        partial_reason = reason or low_error
                        if partial_reason is None:
                            try:
                                row['partial_spearman_r'] = native.partial_spearman_rsa(matrices[model], neural_matrix, low_matrix)
                                row['partial_fisher_z'] = native.fisher_z(row['partial_spearman_r'])
                                row['partial_status'] = 'evaluable'
                            except native.NotEvaluable as exc:
                                row['partial_reason'] = str(exc)
                        else:
                            row['partial_reason'] = partial_reason
                        report['rows'].append(row)
                    del neural_matrix, neural_ranks
                    progress('ROI_observed_rows_recorded', participant=person, held_fold=held, roi=roi)
                folded['complete'] = True
                del matrices, ranks, low_matrix
        check()
        keys = {(row['participant'], row['held_fold'], row['roi'], row['model_label']) for row in report['rows']}
        required = {(person, fold, roi, model) for person in PARTICIPANTS for fold in range(5) for roi in ROIS for model in MODELS}
        require(len(report['rows']) == 400 and keys == required and len(report['folds']) == 25
                and all(fold['complete'] for fold in report['folds']), 'Complete400-row observed support required')
        report['all_400_rows_recorded'] = True
        report['complete_primary_observed_panel'] = all(row[f'{kind}_status'] == 'evaluable'
            for row in report['rows'] if row['roi_role'] == 'primary' for kind in ('native', 'partial'))
        report['complete_EVC_observed_diagnostic'] = all(row['native_status'] == 'evaluable'
            for row in report['rows'] if row['roi_role'] == 'diagnostic')
        report['descriptive_native_effects'] = effect_table(np, native, report['rows'], 'native')
        report['descriptive_partial_effects'] = effect_table(np, native, report['rows'], 'partial')
        check()
        report['complete_observed_panel'] = report['complete_primary_observed_panel'] and report['complete_EVC_observed_diagnostic']
        report['status'] = 'complete_observed_panel' if report['complete_observed_panel'] else 'observed_panel_not_evaluable'
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['complete_observed_panel'] = False
        report['status'] = 'observed_panel_incomplete'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic()-started >= args.max_seconds:
            report['complete_observed_panel'] = False
            report['status'] = 'observed_panel_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_observed_panel'] and time.monotonic()-started >= args.max_seconds:
            report['complete_observed_panel'] = False
            report['status'] = 'observed_panel_incomplete'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_observed_panel'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
