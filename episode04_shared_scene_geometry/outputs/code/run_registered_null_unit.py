#!/usr/bin/env python3
"""One fixed participant/fold and explicitly selected registered null draw ranges.

Only completed observed-panel inputs are eligible. This returns signed component
scores, never a unit-level winner, threshold, candidate decision or partial null.
Disjoint subsequent ranges are selected externally; this process never resumes,
retries, discovers other outputs or silently omits an unavailable draw.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_observed_native_panel as panel


EPISODE = panel.EPISODE
OBSERVED = EPISODE / 'outputs/runs/observed_native_panel/20261005-observed01/observed_native_panel.json'
LOW_LEVEL = EPISODE / 'outputs/runs/source_readiness/20261005-low_level_full01'
MODEL_ORDER = tuple(panel.MODELS)
VLM_ORDER = tuple(panel.VLM_LABELS)
ROI_ORDER = tuple(panel.ROIS[:3])
MAX_SECONDS = 14400  # Prepared ceiling; the selected first retained prefix uses1200.
ENGINEERING_ATOL = 1e-12
REPORT_NAME = 'registered_null_unit.json'


def draw_range(text, limit):
    pieces = text.split(':')
    if len(pieces) != 2 or any(not item.isascii() or not item.isdigit() for item in pieces):
        raise ValueError('Use one explicit half-open START:STOP range')
    start, stop = map(int, pieces)
    if not 0 <= start <= stop <= limit:
        raise ValueError(f'Draw range must lie within0:{limit}')
    return start, stop


def clock_deadline(first_utc, now_utc, now_monotonic, process_deadline):
    first = datetime.fromisoformat(first_utc)
    if first.tzinfo is None or first > now_utc:
        raise ValueError('Original first-panel timestamp must be timezone-aware and not future')
    end = first + timedelta(hours=120)
    remaining = (end-now_utc).total_seconds()
    if remaining <= 0:
        raise TimeoutError('Original120-hour panel budget is exhausted')
    return min(process_deadline, now_monotonic+remaining), end.isoformat()


def factor_geometry(np, native, value):
    """The unchanged helper's fixed input arithmetic, performed once per X."""
    value = native.finite_array(value, 2)
    if value.shape[0] < 3:
        raise ValueError('Random geometry needs at least three evaluation images')
    mean = value.mean(axis=0)
    _, singular, right = np.linalg.svd(value-mean, full_matrices=False)
    return {'mean': mean, 'singular': singular, 'right': right,
            'rows': value.shape[0], 'dimension': value.shape[1],
            'count': min(value.shape[0]-1, value.shape[1])}


def geometry_from_factors(np, native, factors, model_index, draw_index):
    """Literal original centered-Gaussian/QR/sign/reconstruction expressions."""
    if model_index not in range(3) or draw_index not in range(1000):
        raise ValueError('Invalid registered random-geometry index')
    count = factors['count']
    rng = np.random.Generator(np.random.PCG64(100000+1000*model_index+draw_index))
    gaussian = rng.standard_normal((factors['rows'], count))
    gaussian -= gaussian.mean(axis=0)
    orthogonal, triangular = np.linalg.qr(gaussian, mode='reduced')
    signs = np.where(np.diag(triangular) < 0, -1, 1)
    orthogonal *= signs
    raw = factors['mean'] + (orthogonal*factors['singular'][:count]) @ factors['right'][:count]
    return raw, native.l2_rows(raw)


def rank_cache(np, native, model_rdms, neural_rdms):
    """Rank once; a symmetric whole-image relabeling preserves all tied ranks."""
    models = [native.finite_array(matrix, 2) for matrix in model_rdms]
    neural = [native.finite_array(matrix, 2) for matrix in neural_rdms]
    if not models or not neural:
        raise ValueError('Both model and neural geometries are required')
    n = len(models[0])
    if n < 3 or any(matrix.shape != (n, n) for matrix in models+neural):
        raise ValueError('Cached geometries require identical image support')
    # Do not silently symmetrize an input for which rank-permutation equivalence
    # would fail. Native dot-product RDMs and the focused tied fixture are symmetric.
    if any(not np.array_equal(matrix, matrix.T) for matrix in models):
        raise ValueError('Cached model RDM must be exactly symmetric')
    edges = np.triu_indices(n, 1)
    model_ranks = np.array([native.average_ranks(matrix[edges]) for matrix in models])
    neural_ranks = np.array([native.average_ranks(matrix[edges]) for matrix in neural])
    centered_models = model_ranks-model_ranks.mean(axis=1, keepdims=True)
    centered_neural = neural_ranks-neural_ranks.mean(axis=1, keepdims=True)
    model_norms = np.linalg.norm(centered_models, axis=1)
    neural_norms = np.linalg.norm(centered_neural, axis=1)
    if np.any(model_norms == 0) or np.any(neural_norms == 0):
        raise native.NotEvaluable('Constant null geometry is not evaluable')
    symmetric = np.zeros((len(models), n, n), dtype=np.float64)
    symmetric[:, edges[0], edges[1]] = centered_models
    symmetric[:, edges[1], edges[0]] = centered_models
    return {'symmetric_centered_model_ranks': symmetric, 'centered_neural_ranks': centered_neural,
            'denominators': model_norms[:, None]*neural_norms[None, :],
            'neural_ranks': neural_ranks, 'edges': edges, 'rows': n}


def permutation_scores(np, native, cache, strata, draw_index):
    if draw_index not in range(10000) or len(strata) != cache['rows']:
        raise ValueError('Invalid permutation index or image support')
    permutation = native.whole_image_permutation(strata, draw_index)
    left, right = cache['edges']
    # One bijection serves every model and every sector. No edge is independently
    # shuffled, and no new sort is needed: all upper-triangle pairs remain present.
    moved = cache['symmetric_centered_model_ranks'][:, permutation[left], permutation[right]]
    correlations = np.clip((moved @ cache['centered_neural_ranks'].T)/cache['denominators'], -1, 1)
    z = np.array([[native.fisher_z(float(value)) for value in row] for row in correlations])
    return correlations, z


def validate_observed(record, path):
    panel.require(path == OBSERVED, 'Only the fixed completed observed01 report is selected')
    panel.require(record.get('analysis_id') == 'fixed_panel'
                  and record.get('mode') == 'complete_observed_native_and_low_level_partial_RSA'
                  and record.get('status') == 'complete_observed_panel'
                  and all(record.get(key) is True for key in (
                      'complete_observed_panel', 'all_400_rows_recorded', 'complete_primary_observed_panel',
                      'complete_EVC_observed_diagnostic', 'full_feature_coverage_qualified_before_scores'))
                  and record.get('errors') == [] and record.get('null_draws') == 0
                  and all(record.get(key) is False for key in (
                      'candidate_selection_performed', 'candidate_lock_authorized',
                      'primary_support_pruned_for_EVC', 'preprocessing_or_NC_refitted'))
                  and Path(record['low_level_report']).resolve() == LOW_LEVEL/'low_level_preparation_report.json',
                  'Complete unchanged observed panel required before null draws')
    required = {(person, fold, roi, model) for person in panel.PARTICIPANTS
                for fold in range(5) for roi in panel.ROIS for model in MODEL_ORDER}
    rows = record['rows']
    keyed = {(row['participant'], row['held_fold'], row['roi'], row['model_label']): row for row in rows}
    panel.require(len(rows) == len(keyed) == 400 and set(keyed) == required,
                  'Exactly400 unique observed rows required')
    folds = record['folds']
    panel.require(len(folds) == 25 and {(row['participant'], row['held_fold']) for row in folds}
                  == {(person, fold) for person in panel.PARTICIPANTS for fold in range(5)}
                  and all(row.get('complete') is True for row in folds), 'Exactly25 complete folds required')
    for person in panel.PARTICIPANTS:
        panel.require(Path(record['preprocessing_reports'][person]).resolve()
                      == panel.saved.preprocessing_report(person).resolve()
                      and Path(record['training_context_reports'][person]).resolve()
                      == panel.saved.CONTEXT_REPORTS[person].resolve(), 'Observed source bindings differ')
    return keyed


def load_unit(np, native, low_module, observed, observed_path, person, held, report, check):
    keyed = validate_observed(observed, observed_path)
    expected, feature_index, declarations, features, low_index, _low_features = panel.complete_features(
        np, low_module, LOW_LEVEL, check)
    images, folds = panel.participant_inputs(np, person, expected, check)
    fold = next(item for item in folds if item['held_fold'] == held)
    panel.require(fold['primary_reason'] is None, 'Fixed primary evaluation support is unavailable')
    indices, directory = fold['indices'], fold['directory']
    old = next(item for item in observed['folds'] if item['participant'] == person and item['held_fold'] == held)
    identity_path = observed_path.parent / f'{person}-fold-{held}-evaluation_images.jsonl'
    panel.require(Path(old['image_id_file']).resolve() == identity_path
                  and old['evaluation_images'] == len(indices)
                  and old['source_counts'] == fold['source_counts']
                  and old['EVC_missing_image_ids'] == fold['evc_missing'], 'Observed fold support differs')
    records = []
    for index in indices:
        image, validity = images[int(index)], fold['validity'][int(index)]
        records.append({'image_id': image['image_id'], 'pattern_row': int(index),
            'embedding_row': feature_index[image['image_id']]['output_row'],
            'low_level_row': low_index[image['image_id']], 'source_stratum': image['source_stratum'],
            'component_id': image['component_id'], 'primary_valid_repeats': validity['primary_valid_repeats'],
            'evc_valid_repeats': validity['evc_valid_repeats']})
    panel.require(panel.saved.json_lines(identity_path) == records, 'Exact completed evaluation image list required')
    feature_rows = np.array([row['embedding_row'] for row in records], dtype=np.int64)
    raw = {model: np.asarray(features[model][feature_rows, :], dtype=np.float64) for model in MODEL_ORDER}
    model_rdms = [native.rdm(raw[model], 'cosine') for model in MODEL_ORDER]
    neural_rdms = []
    for roi in ROI_ORDER:
        check()
        voxels = np.load(directory/f'{roi}_voxel_indices_c.npy', allow_pickle=False)
        trained = fold['training']['rois'][roi]
        panel.require(voxels.ndim == 1 and voxels.dtype.kind in 'iu' and len(set(voxels.tolist())) == len(voxels)
                      and len(voxels) == fold['original']['roi_support'][roi]['retained_voxels']
                      == trained['retained_voxels'] and len(voxels) >= 3, 'Frozen ROI support differs')
        with np.load(Path(trained['per_voxel_values']), allow_pickle=False) as values:
            panel.require(np.array_equal(values['voxel_indices_c'], voxels), 'Training context voxel IDs differ')
        patterns = np.load(directory/f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
        panel.require(patterns.shape == (4712, len(voxels)) and patterns.dtype == np.dtype('float64'),
                      'Frozen mean-pattern header differs')
        neural_rdms.append(native.rdm(patterns[indices, :], 'correlation'))
        del patterns
        for model in MODEL_ORDER:
            row = keyed[(person, held, roi, model)]
            panel.require(row['model_id'] == panel.MODELS[model][1] and row['native_status'] == 'evaluable'
                          and row['evaluation_images'] == len(indices) and row['retained_voxels'] == len(voxels)
                          and Path(row['evaluation_image_id_file']).resolve() == identity_path,
                          'Observed row identity/support differs')
    cache = rank_cache(np, native, model_rdms, neural_rdms)
    maximum = 0.
    fixed = {}
    for m, model in enumerate(MODEL_ORDER):
        fixed[model] = []
        for r, roi in enumerate(ROI_ORDER):
            old_row = keyed[(person, held, roi, model)]
            correlation = native.spearman_rsa(model_rdms[m], neural_rdms[r])
            z = native.fisher_z(correlation)
            for actual, prior in ((correlation, old_row['native_spearman_r']), (z, old_row['native_fisher_z'])):
                panel.require(type(prior) in (int, float) and math.isfinite(prior), 'Nonfinite saved observed score')
                maximum = max(maximum, abs(actual-prior))
            fixed[model].append(z)
    panel.require(maximum <= ENGINEERING_ATOL, 'Observed same-support score replay differs beyond1e-12')
    report.update({'source_context_validated': True, 'evaluation_images': len(records),
        'image_order': [row['image_id'] for row in records], 'source_counts': fold['source_counts'],
        'evaluation_image_file': str(identity_path), 'preprocessing_report': str(panel.saved.preprocessing_report(person)),
        'training_context_report': str(panel.saved.CONTEXT_REPORTS[person]),
        'models': [dict(model_label=model, **declarations[model]['source_declarations']) for model in MODEL_ORDER],
        'observed_replay_maximum_absolute_discrepancy': maximum, 'observed_replay_atol': ENGINEERING_ATOL,
        'fixed_observed_fisher_z': fixed})
    return raw, cache, [row['source_stratum'] for row in records]


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', choices=panel.PARTICIPANTS, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--permutation-range', required=True)
    parser.add_argument('--geometry-range', required=True)
    parser.add_argument('--observed-report', type=Path, required=True)
    parser.add_argument('--low-level-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    host = socket.gethostname()
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in host or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(name) != '1' for name in panel.saved.THREAD_ENV)):
        raise RuntimeError('Use one actual compute CPU and six numerical thread limits=1')
    permutation_range = draw_range(args.permutation_range, 10000)
    geometry_range = draw_range(args.geometry_range, 1000)
    panel.require(permutation_range[0] < permutation_range[1] or geometry_range[0] < geometry_range[1],
                  'Select at least one registered draw; no checking-only invocation')
    panel.require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
                  'Use EP04 cwd and a positive explicit process cap<=14400s')
    output, observed_path = args.output_dir.resolve(), args.observed_report.resolve()
    panel.require(not output.exists() and output.is_relative_to(EPISODE/'outputs/runs/registered_nulls')
                  and observed_path == OBSERVED and args.low_level_dir.resolve() == LOW_LEVEL,
                  'Fresh registered-null output and exact observed01/full01 inputs required')
    output.mkdir(parents=True)
    deadline_at = started+args.max_seconds
    report = {'analysis_id': 'controls_diagnostics', 'mode': 'registered_null_participant_fold_unit',
        'status': 'started', 'complete_selected_ranges': False, 'complete_registered_nulls': False,
        'participant': args.participant, 'held_fold': args.fold, 'job_id': os.environ['SLURM_JOB_ID'], 'hostname': host,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'internal_seconds': args.max_seconds,
        'observed_report': str(observed_path), 'low_level_report': str(LOW_LEVEL/'low_level_preparation_report.json'),
        'source_context_validated': False, 'permutation_range': list(permutation_range), 'geometry_range': list(geometry_range),
        'model_order': list(MODEL_ORDER), 'random_VLM_order': list(VLM_ORDER), 'ROI_order': list(ROI_ORDER),
        'registered_permutation_draws': 10000, 'registered_geometry_draws_per_VLM': 1000,
        'random_X_scope': 'Exact eligible evaluation-fold raw release embedding matrix before downstream L2',
        'permutation_unit': 'Whole image within registered source stratum; same bijection for all models and sectors',
        'random_seed': 'PCG64(100000+1000*model_index+draw_index)', 'permutation_seed': 'PCG64(draw_index)',
        'signed_finite_null_values_preserved': True, 'fixed_DINO_for_geometry_null': True,
        'local_winner_or_threshold_computed': False, 'candidate_selection_performed': False,
        'candidate_lock_authorized': False, 'new_preprocessing_or_neural_fit': False,
        'raw_beta_arrays_or_images_opened': False, 'excluded_outcomes_opened': False,
        'automatic_retry': False, 'automatic_resume': False, 'external_range_selection_only': True,
        'score_file': 'draw_scores.jsonl', 'completed_permutation_draws': 0,
        'completed_geometry_by_model': {model: 0 for model in VLM_ORDER}, 'errors': []}

    def check():
        if time.monotonic() >= deadline_at:
            raise TimeoutError('Selected process bound or original120-hour panel budget reached')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output/'registered_null_unit.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/REPORT_NAME)

    def progress(stage):
        check()
        report['stage'] = stage
        save()
        print(json.dumps({'stage': stage, 'elapsed_seconds': report['elapsed_seconds'],
                          'completed_permutation_draws': report['completed_permutation_draws'],
                          'completed_geometry_by_model': report['completed_geometry_by_model']}), flush=True)
        check()

    def interrupted(*_):
        raise TimeoutError('Registered null unit interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_at-time.monotonic()))
        progress('before_completed_observed_input')
        observed = panel.read_json(observed_path)
        validate_observed(observed, observed_path)
        deadline_at, wall_end = clock_deadline(observed['first_panel_evaluation_utc'], datetime.now(timezone.utc),
                                             time.monotonic(), deadline_at)
        report['first_panel_evaluation_utc'] = observed['first_panel_evaluation_utc']
        report['panel_wall_deadline_utc'] = wall_end
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline_at-time.monotonic()))
        progress('before_exact_unit_inputs')
        import numpy as np
        import native_rsa as native
        import low_level_pixels_proposal as low_module
        raw, cache, strata = load_unit(np, native, low_module, observed, observed_path,
                                      args.participant, args.fold, report, check)
        factors = {}
        if geometry_range[0] < geometry_range[1]:
            for model in VLM_ORDER:
                progress(f'fixed_SVD_{model}')
                factors[model] = factor_geometry(np, native, raw[model])
        report['setup_seconds'] = time.monotonic()-started
        progress('registered_draws')
        with (output/'draw_scores.jsonl').open('x') as stream:
            for draw in range(*permutation_range):
                check()
                report['current_attempt'] = {'family': 'whole_image_permutation', 'draw_index': draw}
                before = time.monotonic()
                correlations, z = permutation_scores(np, native, cache, strata, draw)
                record = {'family': 'whole_image_permutation', 'draw_index': draw,
                          'model_order': list(MODEL_ORDER), 'ROI_order': list(ROI_ORDER),
                          'spearman_r': correlations.tolist(), 'fisher_z': z.tolist(),
                          'elapsed_seconds': time.monotonic()-before}
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                report['completed_permutation_draws'] += 1
                report['current_attempt'] = None
                if report['completed_permutation_draws'] % 64 == 0:
                    progress('registered_permutation_prefix_retained')
                check()
            for draw in range(*geometry_range):
                for model_index, model in enumerate(VLM_ORDER):
                    check()
                    report['current_attempt'] = {'family': 'covariance_matched_random_geometry',
                                                 'draw_index': draw, 'model_label': model}
                    before = time.monotonic()
                    _raw, normalized = geometry_from_factors(np, native, factors[model], model_index, draw)
                    generation_seconds = time.monotonic()-before
                    check()
                    matrix = 1-normalized @ normalized.T
                    np.fill_diagonal(matrix, 0)
                    ranks = native.average_ranks(native.upper_triangle(matrix))
                    correlations = [native.pearson(ranks, target) for target in cache['neural_ranks']]
                    z = [native.fisher_z(value) for value in correlations]
                    record = {'family': 'covariance_matched_random_geometry', 'draw_index': draw,
                              'model_label': model, 'model_index': model_index,
                              'seed': 100000+1000*model_index+draw, 'ROI_order': list(ROI_ORDER),
                              'spearman_r': correlations, 'fisher_z': z,
                              'generation_seconds': generation_seconds,
                              'RDM_rank_score_seconds': time.monotonic()-before-generation_seconds,
                              'elapsed_seconds': time.monotonic()-before}
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                    stream.flush()
                    report['completed_geometry_by_model'][model] += 1
                    report['current_attempt'] = None
                    del _raw, normalized, matrix, ranks
                    progress('registered_geometry_component_retained')
        check()
        panel.require(report['completed_permutation_draws'] == permutation_range[1]-permutation_range[0]
                      and all(value == geometry_range[1]-geometry_range[0]
                              for value in report['completed_geometry_by_model'].values()),
                      'Every selected draw/component must be retained')
        report['complete_selected_ranges'] = True
        report['status'] = 'completed_selected_null_ranges'
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['complete_selected_ranges'] = False
        report['status'] = 'incomplete_selected_null_ranges'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline_at:
            report['complete_selected_ranges'] = False
            report['status'] = 'incomplete_selected_null_ranges'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_selected_ranges'] and time.monotonic() >= deadline_at:
            report['complete_selected_ranges'] = False
            report['status'] = 'incomplete_selected_null_ranges'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_selected_ranges'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
