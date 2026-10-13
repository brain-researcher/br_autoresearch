#!/usr/bin/env python3
"""One frozen EP04 participant/fold: secondary ridge encoding on RSA support.

Normalized loss, lambda=.001, training-only means and an unpenalized intercept.
Saved development patterns only; no preprocessing, raw images/betas, tuning,
primary selection, automatic follow-on or candidate-lock decision.
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

import run_observed_native_panel as observed

EPISODE = observed.EPISODE
MAX_SECONDS = 1200
LAMBDA = .001
BLOCK_VOXELS = 256
REPORT_NAME = 'identical_fold_encoding.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_ridge(training, evaluation):
    """Inputs are native row-L2 embeddings; only training enters the fit."""
    import numpy as np
    from scipy.linalg import cho_factor
    x = np.asarray(training, dtype=np.float64)
    test = np.asarray(evaluation, dtype=np.float64)
    require(x.ndim == test.ndim == 2 and len(x) >= 2 and len(test) >= 1
            and x.shape[1] == test.shape[1] and x.shape[1] > 0
            and np.isfinite(x).all() and np.isfinite(test).all(),
            'Finite matching training/evaluation designs required')
    mean = x.mean(axis=0)
    centered = x - mean
    gram = centered.T @ centered
    gram.flat[::gram.shape[0]+1] += len(x) * LAMBDA
    return {'training_centered': centered, 'evaluation_centered': test-mean,
            'training_mean': mean, 'factor': cho_factor(gram, lower=True, check_finite=True),
            'training_images': len(x)}


def predict_block(design, training_responses):
    """Each voxel block gets its own training-mean intercept and ridge weights."""
    import numpy as np
    from scipy.linalg import cho_solve
    y = np.asarray(training_responses, dtype=np.float64)
    require(y.ndim == 2 and len(y) == design['training_images'] and y.shape[1] > 0
            and np.isfinite(y).all(), 'Finite training response block required')
    mean = y.mean(axis=0)
    weights = cho_solve(design['factor'], design['training_centered'].T @ (y-mean),
                        check_finite=True)
    return design['evaluation_centered'] @ weights + mean, mean


def squared_errors(evaluation, prediction, training_mean):
    import numpy as np
    y, predicted, mean = (np.asarray(value, dtype=np.float64)
                          for value in (evaluation, prediction, training_mean))
    require(y.ndim == 2 and y.shape == predicted.shape and mean.shape == (y.shape[1],),
            'Evaluation/prediction/baseline shapes differ')
    if not all(np.isfinite(value).all() for value in (y, predicted, mean)):
        return None, None
    sse = float(np.sum(np.square(y-predicted), dtype=np.float64))
    baseline = float(np.sum(np.square(y-mean), dtype=np.float64))
    return (sse, baseline) if math.isfinite(sse) and math.isfinite(baseline) else (None, None)


def finish_score(sse, baseline):
    """No test-mean centering, clipping of negative R², or epsilon baseline."""
    if not math.isfinite(sse) or not math.isfinite(baseline) or baseline <= 0:
        return None
    value = 1-sse/baseline
    return value if math.isfinite(value) else None


def support_indices(images, validity, held, evc=False):
    training = [i for i, (image, row) in enumerate(zip(images, validity))
                if image['fold'] != held and row['primary_eligible']
                and (not evc or row['evc_eligible_on_primary_images'])]
    evaluation = [i for i, (image, row) in enumerate(zip(images, validity))
                  if image['fold'] == held and row['primary_eligible']]
    missing = [i for i in evaluation if evc and not validity[i]['evc_eligible_on_primary_images']]
    return training, evaluation, missing


def observed_support(path, report, low_dir, person, fold, images, embedding_index, low_index):
    """Match identities only; observed effect sizes never configure this fit."""
    require(report.get('status') in ('complete_observed_panel', 'observed_panel_not_evaluable')
            and report.get('all_400_rows_recorded') is True and report.get('errors') == []
            and report.get('full_feature_coverage_qualified_before_scores') is True
            and report.get('candidate_selection_performed') is False
            and report.get('candidate_lock_authorized') is False
            and report.get('primary_support_pruned_for_EVC') is False
            and report.get('preprocessing_or_NC_refitted') is False
            and Path(report['low_level_report']).resolve() == low_dir / 'low_level_preparation_report.json',
            'Completed unchanged observed panel and the same full feature handoff required')
    required = {(p, f, roi, model) for p in observed.PARTICIPANTS for f in range(5)
                for roi in observed.ROIS for model in observed.MODELS}
    keys = [(r['participant'], r['held_fold'], r['roi'], r['model_label']) for r in report['rows']]
    require(len(keys) == 400 and set(keys) == required, 'Observed panel has missing/duplicate rows')
    folds = report['folds']
    require(len(folds) == 25 and {(f['participant'], f['held_fold']) for f in folds}
            == {(p, h) for p in observed.PARTICIPANTS for h in range(5)}
            and all(f['complete'] is True for f in folds), 'Observed folds are incomplete')
    for key, expected in (('preprocessing_reports', observed.saved.preprocessing_report(person)),
                          ('training_context_reports', observed.saved.CONTEXT_REPORTS[person])):
        require(Path(report[key][person]).resolve() == expected.resolve(), 'Observed source context differs')
    held = fold['held_fold']
    identity = path.parent / f'{person}-fold-{held}-evaluation_images.jsonl'
    selected = next(f for f in folds if f['participant'] == person and f['held_fold'] == held)
    require(Path(selected['image_id_file']).resolve() == identity
            and selected['evaluation_images'] == len(fold['indices'])
            and selected['source_counts'] == fold['source_counts']
            and selected['EVC_missing_image_ids'] == fold['evc_missing'],
            'Observed selected-fold support differs')
    expected_rows = []
    for index in fold['indices']:
        i = int(index)
        image, validity = images[i], fold['validity'][i]
        expected_rows.append({'image_id': image['image_id'], 'pattern_row': i,
            'embedding_row': embedding_index[image['image_id']]['output_row'],
            'low_level_row': low_index[image['image_id']], 'source_stratum': image['source_stratum'],
            'component_id': image['component_id'], 'primary_valid_repeats': validity['primary_valid_repeats'],
            'evc_valid_repeats': validity['evc_valid_repeats']})
    require(observed.saved.json_lines(identity) == expected_rows,
            'Encoding evaluation IDs must exactly equal the saved primary RSA IDs')
    selected_rows = {(r['roi'], r['model_label']): r for r in report['rows']
                     if r['participant'] == person and r['held_fold'] == held}
    for (roi, model), row in selected_rows.items():
        require(row['model_id'] == observed.MODELS[model][1]
                and row['evaluation_images'] == len(expected_rows)
                and Path(row['evaluation_image_id_file']).resolve() == identity
                and row['retained_voxels'] == fold['original']['roi_support'][roi]['retained_voxels'],
                'Observed model/ROI support differs')
    return identity, report.get('job_id')


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant', choices=observed.PARTICIPANTS, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--low-level-dir', type=Path, required=True)
    parser.add_argument('--observed-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    hostname = socket.gethostname()
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in hostname
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(key) == '1' for key in observed.saved.THREAD_ENV),
            'One actual compute CPU and six numerical thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
            'Use EP04 cwd and explicit whole-process bound<=1200s')
    output, low_dir, observed_path = (p.resolve() for p in
                                     (args.output_dir, args.low_level_dir, args.observed_report))
    require(not output.exists() and output.is_relative_to(EPISODE / 'outputs/runs/identical_fold_encoding')
            and low_dir.is_relative_to(EPISODE / 'outputs/runs/source_readiness')
            and observed_path.is_relative_to(EPISODE / 'outputs/runs/observed_native_panel')
            and observed_path.name == 'observed_native_panel.json',
            'Fresh encoding output and existing episode-local feature/observed inputs required')
    output.mkdir(parents=True)
    report = {'analysis_id': 'controls_diagnostics', 'diagnostic': 'identical_fold_encoding',
        'participant': args.participant,
        'held_fold': args.fold, 'job_id': os.environ['SLURM_JOB_ID'], 'hostname': hostname,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'internal_seconds': args.max_seconds,
        'status': 'started', 'all_16_rows_recorded': False, 'complete_encoding_diagnostic': False,
        'normalized_loss_lambda': LAMBDA, 'intercept': 'unpenalized_training_mean',
        'model_input': 'row_L2_native_embedding', 'voxel_block_size': BLOCK_VOXELS,
        'evaluation_support': 'exact_primary_RSA_IDs_without_EVC_pruning',
        'low_level_report': str(low_dir / 'low_level_preparation_report.json'),
        'observed_report': str(observed_path), 'raw_beta_arrays_read': False,
        'raw_images_or_excluded_outcomes_opened': False, 'preprocessing_or_NC_refitted': False,
        'candidate_selection_performed': False, 'candidate_lock_authorized': False,
        'lambda_or_rank_search_performed': False, 'automatic_retry': False, 'automatic_resume': False,
        'interpretation': 'Secondary development readout; disagreement cannot select or rescue native RSA',
        'rows': [{'participant': args.participant, 'held_fold': args.fold, 'roi': roi,
                  'roi_role': 'primary' if roi in observed.ROIS[:3] else 'diagnostic',
                  'model_label': model, 'model_id': observed.MODELS[model][1],
                  'status': 'not_started', 'complete': False, 'processed_voxels': 0,
                  'SSE_partial': 0., 'SSE_baseline_partial': 0., 'R2_SSE': None}
                 for model in observed.MODELS for roi in observed.ROIS], 'errors': []}

    panel_deadline = None

    def deadline_reached():
        return (time.monotonic()-started >= args.max_seconds
                or (panel_deadline is not None and datetime.now(timezone.utc) >= panel_deadline))

    def check():
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('Identical-fold encoding whole-process deadline reached')
        if panel_deadline is not None and datetime.now(timezone.utc) >= panel_deadline:
            raise TimeoutError('Frozen 120-hour panel clock expired')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output / 'identical_fold_encoding.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output / REPORT_NAME)

    def progress(phase):
        report['phase'] = phase
        save()
        check()

    def fail(exc):
        report['complete_encoding_diagnostic'] = False
        report['status'] = 'technical_encoding_failure'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})

    def interrupted(*_):
        raise TimeoutError('Identical-fold encoding interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.max_seconds-(time.monotonic()-started)))
        progress('qualify_saved_feature_and_neural_support')
        observed_report = observed.read_json(observed_path)
        first_score = datetime.fromisoformat(observed_report['first_panel_evaluation_utc'])
        require(first_score.tzinfo is not None and first_score <= datetime.now(timezone.utc),
                'Observed first-score timestamp must be timezone-aware and not in the future')
        panel_deadline = first_score + timedelta(hours=120)
        report['first_panel_evaluation_utc'] = first_score.isoformat()
        report['panel_clock_deadline_utc'] = panel_deadline.isoformat()
        check()
        signal.setitimer(signal.ITIMER_REAL, max(.001, min(
            args.max_seconds-(time.monotonic()-started),
            (panel_deadline-datetime.now(timezone.utc)).total_seconds())))
        import numpy as np
        import native_rsa as native
        import low_level_pixels_proposal as low_module
        expected, embedding_index, declarations, features, low_index, low_features = observed.complete_features(
            np, low_module, low_dir, check)
        del low_features
        images, folds = observed.participant_inputs(np, args.participant, expected, check)
        fold = folds[args.fold]
        identity, source_job = observed_support(observed_path, observed_report, low_dir, args.participant, fold,
                                               images, embedding_index, low_index)
        report.update(full_feature_coverage_qualified=True, observed_support_matched=True,
                      observed_job_id=source_job, evaluation_image_id_file=str(identity),
                      preprocessing_report=str(observed.saved.preprocessing_report(args.participant)),
                      training_context_report=str(observed.saved.CONTEXT_REPORTS[args.participant]))
        support = {}
        training_manifest = []
        for roi in observed.ROIS:
            train, evaluation, missing = support_indices(images, fold['validity'], args.fold,
                                                         evc=roi == observed.ROIS[3])
            require(evaluation == fold['indices'].tolist(), 'Evaluation support changed')
            require(not ({images[i]['component_id'] for i in train}
                         & {images[i]['component_id'] for i in evaluation}),
                    'Frozen component crosses the training/evaluation boundary')
            support[roi] = (train, evaluation, missing)
        primary_train, evc_train = set(support[observed.ROIS[0]][0]), set(support[observed.ROIS[3]][0])
        for i in sorted(primary_train):
            image = images[i]
            training_manifest.append(dict(image_id=image['image_id'], pattern_row=i,
                embedding_row=embedding_index[image['image_id']]['output_row'], fold=image['fold'],
                component_id=image['component_id'], source_stratum=image['source_stratum'],
                primary_valid_repeats=fold['validity'][i]['primary_valid_repeats'],
                evc_valid_repeats=fold['validity'][i]['evc_valid_repeats'], EVC_training_eligible=i in evc_train))
        training_path = output / 'training_images.jsonl'
        with training_path.open('x') as stream:
            for item in training_manifest:
                stream.write(json.dumps(item)+'\n')
        report['training_image_id_file'] = str(training_path)
        patterns, voxel_counts = {}, {}
        for roi in observed.ROIS:
            check()
            trained = fold['training']['rois'][roi]
            voxels = np.load(fold['directory'] / f'{roi}_voxel_indices_c.npy', allow_pickle=False)
            require(voxels.ndim == 1 and voxels.dtype.kind in 'iu'
                    and len(set(voxels.tolist())) == len(voxels)
                    and len(voxels) == fold['original']['roi_support'][roi]['retained_voxels']
                    and len(voxels) == trained['retained_voxels'], 'ROI voxel support differs')
            with np.load(Path(trained['per_voxel_values']), allow_pickle=False) as stored:
                require(np.array_equal(stored['voxel_indices_c'], voxels), 'Training-context voxel IDs differ')
            array = np.load(fold['directory'] / f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
            require(array.shape == (4712, len(voxels)) and array.dtype == np.dtype('float64'),
                    'Saved mean-pattern header differs')
            patterns[roi], voxel_counts[roi] = array, len(voxels)
        progress('begin_frozen_ridge')
        for model in observed.MODELS:
            designs = {}
            for row in (r for r in report['rows'] if r['model_label'] == model):
                check()
                roi = row['roi']
                train, evaluation, missing = support[roi]
                row.update(training_images=len(train), evaluation_images=len(evaluation),
                           retained_voxels=voxel_counts[roi], evaluation_image_id_file=str(identity),
                           training_image_id_file=str(training_path),
                           EVC_missing_evaluation_image_ids=[images[i]['image_id'] for i in missing])
                reason = ('EVC lacks repeats on required primary IDs; no intersection substitution' if missing
                          else 'Insufficient training/evaluation/voxel support' if len(train) < 2
                          or not evaluation or voxel_counts[roi] == 0 else None)
                if reason is not None:
                    row.update(status='not_evaluable', reason=reason, complete=True)
                    progress('row_not_evaluable')
                    continue
                key = tuple(train)
                if key not in designs:
                    train_rows = [embedding_index[images[i]['image_id']]['output_row'] for i in train]
                    eval_rows = [embedding_index[images[i]['image_id']]['output_row'] for i in evaluation]
                    x_train = native.l2_rows(features[model][train_rows, :])
                    x_eval = native.l2_rows(features[model][eval_rows, :])
                    progress('factor_training_design')
                    designs[key] = prepare_ridge(x_train, x_eval)
                    del x_train, x_eval
                    check()
                row['status'] = 'running'
                progress('fit_and_score_voxel_blocks')
                for start in range(0, voxel_counts[roi], BLOCK_VOXELS):
                    check()
                    columns = np.arange(start, min(start+BLOCK_VOXELS, voxel_counts[roi]))
                    y_train = np.asarray(patterns[roi][np.ix_(train, columns)], dtype=np.float64)
                    y_eval = np.asarray(patterns[roi][np.ix_(evaluation, columns)], dtype=np.float64)
                    check()
                    predicted, mean = predict_block(designs[key], y_train)
                    sse, baseline = squared_errors(y_eval, predicted, mean)
                    if (sse is None or not math.isfinite(row['SSE_partial']+sse)
                            or not math.isfinite(row['SSE_baseline_partial']+baseline)):
                        row.update(status='not_evaluable', reason='Nonfinite response, prediction or squared error', complete=True)
                        progress('row_not_evaluable')
                        break
                    row['SSE_partial'] += sse
                    row['SSE_baseline_partial'] += baseline
                    row['processed_voxels'] += len(columns)
                    progress('voxel_block_complete')
                if row['status'] == 'running':
                    value = finish_score(row['SSE_partial'], row['SSE_baseline_partial'])
                    row.update(SSE=row['SSE_partial'], SSE_baseline=row['SSE_baseline_partial'],
                               R2_SSE=value, complete=True,
                               status='evaluable' if value is not None else 'not_evaluable')
                    if value is None:
                        row['reason'] = 'Zero/nonfinite training-mean baseline or nonfinite R2; no epsilon'
                    progress('row_complete')
            del designs
        report['all_16_rows_recorded'] = all(row['complete'] for row in report['rows'])
        report['complete_encoding_diagnostic'] = all(row['status'] == 'evaluable' for row in report['rows'])
        report['status'] = ('encoding_diagnostic_complete' if report['complete_encoding_diagnostic']
                            else 'encoding_diagnostic_not_evaluable')
        check()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        check()
    except Exception as exc:
        fail(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if deadline_reached() and report['status'] != 'technical_encoding_failure':
            fail(TimeoutError('Deadline reached during finalization'))
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if deadline_reached() and report['status'] != 'technical_encoding_failure':
            fail(TimeoutError('Deadline reached during final publication'))
            save()
    return 0 if report['complete_encoding_diagnostic'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
