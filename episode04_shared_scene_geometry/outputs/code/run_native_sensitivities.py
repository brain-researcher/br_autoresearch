#!/usr/bin/env python3
"""Four registered observed EP04 sensitivities, after the complete native panel.

Reuse frozen evaluation folds, released features, all-valid-repeat means and
training voxel masks. No beta/image read, refit, low-level partial rerun, null,
candidate screening, winner or automatic continuation. EVC is descriptive.
"""
import argparse
from collections import Counter
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
PARTICIPANTS = observed.PARTICIPANTS
ROIS = observed.ROIS
MODELS = observed.MODELS
STRATA = observed.STRATA
SENSITIVITIES = ('near_neighbour_exclusion', 'equal_source_strata',
                 'complete_repeats', 'neural_cosine')
MAX_SECONDS = 2400
require = observed.require


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def admit_observed(path):
    """A complete original panel is required, never a partial-score promotion."""
    require(path.is_relative_to(EPISODE / 'outputs/runs/observed_native_panel'),
            'Only the selected episode-local observed panel is eligible')
    report = observed.read_json(path)
    require(report.get('analysis_id') == 'fixed_panel'
            and report.get('mode') == 'complete_observed_native_and_low_level_partial_RSA'
            and report.get('status') == 'complete_observed_panel'
            and report.get('complete_observed_panel') is True
            and report.get('complete_primary_observed_panel') is True
            and report.get('complete_EVC_observed_diagnostic') is True
            and report.get('all_400_rows_recorded') is True
            and report.get('full_feature_coverage_qualified_before_scores') is True
            and report.get('candidate_selection_performed') is False
            and report.get('candidate_lock_authorized') is False
            and report.get('null_draws') == 0 and report.get('errors') == [],
            'Completed full observed native/primary-partial panel required')
    start = datetime.fromisoformat(report['first_panel_evaluation_utc'])
    finish = datetime.fromisoformat(report['finished_utc'])
    require(start.tzinfo is not None and finish.tzinfo is not None and start <= finish,
            'Observed panel clock/terminal identity required')
    rows = report.get('rows', [])
    keyed = {(r['participant'], r['held_fold'], r['roi'], r['model_label']): r for r in rows}
    expected = {(p, f, r, m) for p in PARTICIPANTS for f in range(5) for r in ROIS for m in MODELS}
    require(len(rows) == len(keyed) == 400 and set(keyed) == expected,
            'All400 exact original panel keys required')
    for key, row in keyed.items():
        require(row['model_id'] == MODELS[key[3]][1]
                and row['roi_role'] == ('primary' if key[2] in ROIS[:3] else 'diagnostic')
                and row.get('native_status') == 'evaluable'
                and finite_number(row.get('native_spearman_r'))
                and abs(row['native_spearman_r']) < 1
                and finite_number(row.get('native_fisher_z')),
                'Every original native row must be finite and correctly bound')
        if key[2] in ROIS[:3]:
            require(row.get('partial_status') == 'evaluable'
                    and finite_number(row.get('partial_spearman_r'))
                    and abs(row['partial_spearman_r']) < 1
                    and finite_number(row.get('partial_fisher_z')),
                    'Every original primary partial row must be finite')
    folds = {(row['participant'], row['held_fold']): row for row in report.get('folds', [])}
    require(len(report.get('folds', [])) == len(folds) == 25
            and set(folds) == {(p, f) for p in PARTICIPANTS for f in range(5)}
            and all(row.get('complete') is True for row in folds.values()),
            'All25 original fold identities required')
    return report, keyed, folds


def bind_base_support(base, base_rows, base_folds, inputs, embedding_index, low_index, check):
    """Match all400 rows and25 exact image lists before sensitivity arithmetic."""
    for person in PARTICIPANTS:
        require(Path(base['preprocessing_reports'][person]).resolve()
                == observed.saved.preprocessing_report(person).resolve()
                and Path(base['training_context_reports'][person]).resolve()
                == observed.saved.CONTEXT_REPORTS[person].resolve(),
                'Observed preprocessing/context source binding differs')
        images, folds = inputs[person]
        for fold in folds:
            check()
            held, indices = fold['held_fold'], fold['indices']
            original = base_folds[(person, held)]
            identity_path = Path(original['image_id_file']).resolve()
            expected = []
            for i in indices:
                i = int(i)
                image, validity = images[i], fold['validity'][i]
                expected.append({'image_id': image['image_id'], 'pattern_row': i,
                    'embedding_row': embedding_index[image['image_id']]['output_row'],
                    'low_level_row': low_index[image['image_id']],
                    'source_stratum': image['source_stratum'], 'component_id': image['component_id'],
                    'primary_valid_repeats': validity['primary_valid_repeats'],
                    'evc_valid_repeats': validity['evc_valid_repeats']})
            require(original['evaluation_images'] == len(indices)
                    and original['source_counts'] == fold['source_counts']
                    and original['EVC_missing_image_ids'] == fold['evc_missing']
                    and identity_path.is_relative_to(Path(base['_report_path']).parent)
                    and observed.saved.json_lines(identity_path) == expected,
                    'Exact original evaluation IDs/repeats/source support differ')
            for row in fold['validity']:
                require(row['complete_four'] is (row['primary_valid_repeats'] >= 4),
                        'Frozen complete-repeat flag differs from >=4 valid presentations')
            counts = Counter(images[int(i)]['source_stratum'] for i in indices
                             if fold['validity'][int(i)]['complete_four'])
            require(all(counts[s] == fold['original']['evaluation_image_counts_by_stratum'][s]['complete_four']
                        for s in STRATA), 'Complete-repeat source counts differ')
            for roi in ROIS:
                trained = fold['training']['rois'][roi]
                for model in MODELS:
                    row = base_rows[(person, held, roi, model)]
                    require(row['evaluation_images'] == len(indices)
                            and Path(row['evaluation_image_id_file']).resolve() == identity_path
                            and row['retained_voxels'] == trained['retained_voxels']
                            == fold['original']['roi_support'][roi]['retained_voxels']
                            and row['training_context_common_images'] == fold['training']['common_training_images']
                            and row['training_repeat_reliability'] == trained['repeat_reliability']
                            and row['training_provider_formula_NC_percent'] == trained['provider_formula_NC_percent'],
                            'Observed400-row voxel/image/training-context support differs')


def frozen_representatives(expected):
    """Global lexical minima before validity, participant or source subsetting."""
    representatives, folds = {}, {}
    for image_id, row in expected.items():
        component = row['component_id']
        if component in folds:
            require(folds[component] == row['fold'], 'Frozen component crosses evaluation folds')
        folds[component] = row['fold']
        representatives[component] = min(image_id, representatives.get(component, image_id))
    return representatives


def selection(np, sensitivity, images, fold, representatives):
    primary = fold['indices']
    if sensitivity == 'near_neighbour_exclusion':
        indices = np.array([int(i) for i in primary
                            if images[int(i)]['image_id'] == representatives[images[int(i)]['component_id']]], dtype=np.int64)
    elif sensitivity == 'complete_repeats':
        indices = np.array([int(i) for i in primary if fold['validity'][int(i)]['complete_four']], dtype=np.int64)
    else:
        indices = primary.copy()
    counts = {s: sum(images[int(i)]['source_stratum'] == s for i in indices) for s in STRATA}
    if sensitivity == 'equal_source_strata':
        cells = [(s, np.array([int(i) for i in indices if images[int(i)]['source_stratum'] == s], dtype=np.int64))
                 for s in STRATA]
    else:
        cells = [('all_sources', indices)]
    return indices, counts, cells


def panel_effects(np, native, rows, sensitivity):
    selected = [row for row in rows if row['sensitivity'] == sensitivity and row['roi_role'] == 'primary']
    if any(row['status'] != 'evaluable' for row in selected):
        return None
    keyed = {(r['model_label'], r['participant'], r['held_fold'], r['roi']): r['fisher_z'] for r in selected}
    def values(model):
        return np.array([[[keyed[(model, person, fold, roi)] for roi in ROIS[:3]]
                          for fold in range(5)] for person in PARTICIPANTS])
    effects = native.panel_effects(np.array([values(model) for model in observed.VLM_LABELS]), values('DINOv2'))
    return {'VLM_order': list(observed.VLM_LABELS), 'participant_order': list(PARTICIPANTS),
            'primary_ROI_order': list(ROIS[:3]), 'fold_order': list(range(5)),
            'screen_performed': False, **{key: value.tolist() for key, value in effects.items()}}


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed-report', type=Path, required=True)
    parser.add_argument('--low-level-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    hostname = socket.gethostname()
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in hostname
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(key) == '1' for key in observed.saved.THREAD_ENV),
            'Use one actual compute CPU with six numerical thread limits=1')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
            'Use EP04 cwd and an explicit whole-process cap<=2400s')
    output, low_dir, source = args.output_dir.resolve(), args.low_level_dir.resolve(), args.observed_report.resolve()
    require(not output.exists() and output.is_relative_to(EPISODE / 'outputs/runs/native_sensitivities')
            and low_dir.is_relative_to(EPISODE / 'outputs/runs/source_readiness'),
            'Fresh episode-local sensitivity output and approved feature handoff required')
    output.mkdir(parents=True)
    report = {'analysis_id': 'controls_diagnostics', 'mode': 'registered_observed_native_sensitivities',
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': hostname,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'started',
              'observed_report': str(source), 'internal_seconds': args.max_seconds,
              'all_1600_rows_recorded': False, 'complete_primary_sensitivities': False,
              'complete_calculation': False, 'base400_support_matched_before_scores': False,
              'sensitivity_order': list(SENSITIVITIES), 'candidate_screen_performed': False,
              'candidate_selection_performed': False, 'candidate_lock_authorized': False,
              'null_draws': 0, 'original_panel_or_partial_RSA_recomputed': False,
              'raw_beta_or_image_values_read': False, 'preprocessing_or_NC_refitted': False,
              'EVC_descriptive_only': True, 'primary_support_pruned_for_EVC': False,
              'automatic_retry': False, 'automatic_resume': False,
              'repeat_means': 'Existing mean of all valid presentations, never truncated to four',
              'near_neighbours': 'One global lexical image-ID minimum per frozen development component before validity; no replacement',
              'source_weighting': 'Separate within-source Spearman RSA, then equal mean of three Fisher-z cells; no cross-source edges',
              'reliability_context': 'Unchanged training-only context; not refitted to sensitivity/evaluation subsets',
              'folds': [], 'rows': [], 'errors': []}

    deadline = started + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Native sensitivity process or original120-hour panel deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output / 'native_sensitivities.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output / 'native_sensitivities.json')

    def progress(phase, **fields):
        check()
        report['phase'] = phase
        save()
        print(json.dumps({'phase': phase, 'rows_recorded': len(report['rows']),
                          'elapsed_seconds': report['elapsed_seconds'], **fields}), flush=True)
        check()

    def interrupted(*_):
        raise TimeoutError('Native sensitivity interrupted or deadline reached')

    def incomplete(error):
        report['complete_calculation'] = False
        report['complete_primary_sensitivities'] = False
        report['status'] = 'native_sensitivities_incomplete'
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, args.max_seconds-(time.monotonic()-started)))
        progress('admit_completed_observed_panel')
        base, base_rows, base_folds = admit_observed(source)
        panel_started = datetime.fromisoformat(base['first_panel_evaluation_utc'])
        now = datetime.now(timezone.utc)
        require(panel_started <= now and datetime.fromisoformat(base['finished_utc']) <= now,
                'Observed panel timestamps cannot be in the future')
        panel_deadline = panel_started + timedelta(hours=120)
        remaining_panel_seconds = (panel_deadline-now).total_seconds()
        deadline = min(deadline, time.monotonic()+remaining_panel_seconds)
        report['original_panel_deadline_utc'] = panel_deadline.isoformat()
        report['panel_seconds_remaining_at_admission'] = remaining_panel_seconds
        check()
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        require(Path(base['low_level_report']).resolve() == low_dir / 'low_level_preparation_report.json',
                'Sensitivity and original panel low-level source identity differ')
        base['_report_path'] = str(source)
        import numpy as np
        import native_rsa as native
        import low_level_pixels_proposal as low_module
        expected, embedding_index, declarations, features, low_index, low_features = observed.complete_features(np, low_module, low_dir, check)
        inputs = {person: observed.participant_inputs(np, person, expected, check) for person in PARTICIPANTS}
        bind_base_support(base, base_rows, base_folds, inputs, embedding_index, low_index, check)
        del low_features
        representatives = frozen_representatives(expected)
        report['base400_support_matched_before_scores'] = True
        report['original_first_panel_evaluation_utc'] = base['first_panel_evaluation_utc']
        report['original_observed_finished_utc'] = base['finished_utc']
        report['models'] = [dict(model_label=model, **declarations[model]['source_declarations']) for model in MODELS]
        report['global_frozen_components'] = len(representatives)
        representative_file = output / 'global_component_representatives.json'
        representative_file.write_text(json.dumps(representatives, sort_keys=True, indent=2)+'\n')
        report['global_representative_file'] = str(representative_file)
        progress('begin_registered_sensitivities')
        for person in PARTICIPANTS:
            images, folds = inputs[person]
            for fold in folds:
                held, directory = fold['held_fold'], fold['directory']
                patterns, voxel_ids = {}, {}
                for roi in ROIS:
                    check()
                    voxels = np.load(directory / f'{roi}_voxel_indices_c.npy', allow_pickle=False)
                    trained = fold['training']['rois'][roi]
                    require(voxels.ndim == 1 and voxels.dtype.kind in 'iu'
                            and len(set(voxels.tolist())) == len(voxels)
                            and len(voxels) == trained['retained_voxels'], 'ROI voxel identity differs')
                    with np.load(Path(trained['per_voxel_values']), allow_pickle=False) as stored:
                        require(np.array_equal(stored['voxel_indices_c'], voxels), 'Context voxel order differs')
                    patterns[roi] = np.load(directory / f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
                    require(patterns[roi].shape == (4712, len(voxels))
                            and patterns[roi].dtype == np.dtype('float64'), 'Mean-pattern shape/dtype differs')
                    voxel_ids[roi] = voxels
                for sensitivity in SENSITIVITIES:
                    indices, counts, cells = selection(np, sensitivity, images, fold, representatives)
                    identity_path = output / f'{person}-fold-{held}-{sensitivity}-images.jsonl'
                    with identity_path.open('x') as stream:
                        for i in indices:
                            i = int(i)
                            image, validity = images[i], fold['validity'][i]
                            stream.write(json.dumps({'image_id': image['image_id'], 'pattern_row': i,
                                'embedding_row': embedding_index[image['image_id']]['output_row'],
                                'source_stratum': image['source_stratum'], 'component_id': image['component_id'],
                                'primary_valid_repeats': validity['primary_valid_repeats'],
                                'evc_valid_repeats': validity['evc_valid_repeats'],
                                'intended_repeats': validity['intended_repeats'],
                                'observed_metadata_repeats': validity['observed_metadata_repeats']})+'\n')
                    folded = {'participant': person, 'held_fold': held, 'sensitivity': sensitivity,
                              'original_primary_images': len(fold['indices']), 'evaluation_images': len(indices),
                              'source_counts': counts, 'image_id_file': str(identity_path), 'complete': False,
                              'EVC_missing_image_ids': [images[int(i)]['image_id'] for i in indices
                                  if not fold['validity'][int(i)]['evc_eligible_on_primary_images']]}
                    report['folds'].append(folded)
                    scored = {(roi, model): [] for roi in ROIS for model in MODELS}
                    for stratum, cell_indices in cells:
                        progress('sensitivity_geometry', participant=person, held_fold=held,
                                 sensitivity=sensitivity, source_cell=stratum)
                        reason = None
                        if len(cell_indices) < 3 or (stratum == 'all_sources' and min(counts.values()) < 3):
                            reason = 'Fewer than three images in a required fixed fold/source cell'
                        model_ranks, model_errors = {}, {}
                        feature_rows = np.array([embedding_index[images[int(i)]['image_id']]['output_row']
                                                 for i in cell_indices], dtype=np.int64)
                        for model in MODELS:
                            check()
                            try:
                                if reason:
                                    raise native.NotEvaluable(reason)
                                matrix = native.rdm(features[model][feature_rows, :], 'cosine')
                                model_ranks[model] = native.average_ranks(native.upper_triangle(matrix))
                                del matrix
                            except native.NotEvaluable as exc:
                                model_errors[model] = str(exc)
                        for roi in ROIS:
                            check()
                            neural_error, neural_ranks = reason, None
                            if len(voxel_ids[roi]) < 3:
                                neural_error = 'Fewer than three frozen training-qualified ROI voxels'
                            if roi == ROIS[3] and any(not fold['validity'][int(i)]['evc_eligible_on_primary_images'] for i in cell_indices):
                                neural_error = 'EVC lacks repeats on required sensitivity IDs; no favorable intersection'
                            try:
                                if neural_error:
                                    raise native.NotEvaluable(neural_error)
                                matrix = native.rdm(patterns[roi][cell_indices, :],
                                    'cosine' if sensitivity == 'neural_cosine' else 'correlation')
                                neural_ranks = native.average_ranks(native.upper_triangle(matrix))
                                del matrix
                            except native.NotEvaluable as exc:
                                neural_error = str(exc)
                            for model in MODELS:
                                check()
                                result = {'source_cell': stratum, 'evaluation_images': len(cell_indices),
                                          'status': 'not_evaluable', 'spearman_r': None, 'fisher_z': None}
                                error = neural_error or model_errors.get(model)
                                try:
                                    if error:
                                        raise native.NotEvaluable(error)
                                    result['spearman_r'] = native.pearson(model_ranks[model], neural_ranks)
                                    result['fisher_z'] = native.fisher_z(result['spearman_r'])
                                    result['status'] = 'evaluable'
                                except native.NotEvaluable as exc:
                                    result['reason'] = str(exc)
                                scored[(roi, model)].append(result)
                            del neural_ranks
                        del model_ranks
                    for roi in ROIS:
                        trained = fold['training']['rois'][roi]
                        for model in MODELS:
                            values = scored[(roi, model)]
                            evaluable = all(item['status'] == 'evaluable' for item in values)
                            aggregate_z = float(np.mean([item['fisher_z'] for item in values])) if evaluable else None
                            report['rows'].append({'participant': person, 'held_fold': held, 'roi': roi,
                                'roi_role': 'primary' if roi in ROIS[:3] else 'diagnostic',
                                'model_label': model, 'model_id': MODELS[model][1], 'sensitivity': sensitivity,
                                'evaluation_images': len(indices), 'evaluation_image_id_file': str(identity_path),
                                'source_counts': counts, 'retained_voxels': len(voxel_ids[roi]),
                                'neural_distance': 'cosine' if sensitivity == 'neural_cosine' else 'correlation',
                                'aggregation': 'equal_mean_three_source_Fisher_z' if sensitivity == 'equal_source_strata' else 'single_whole_upper_triangle',
                                'status': 'evaluable' if evaluable else 'not_evaluable',
                                'fisher_z': aggregate_z, 'spearman_r': values[0]['spearman_r'] if len(values) == 1 else None,
                                'source_cell_results': values,
                                'training_context_common_images': fold['training']['common_training_images'],
                                'training_repeat_reliability': trained['repeat_reliability'],
                                'training_provider_formula_NC_percent': trained['provider_formula_NC_percent']})
                    folded['complete'] = True
                    progress('sensitivity_fold_recorded', participant=person, held_fold=held, sensitivity=sensitivity)
                del patterns, voxel_ids
        check()
        expected_keys = {(s, p, f, r, m) for s in SENSITIVITIES for p in PARTICIPANTS
                         for f in range(5) for r in ROIS for m in MODELS}
        actual_keys = {(r['sensitivity'], r['participant'], r['held_fold'], r['roi'], r['model_label']) for r in report['rows']}
        require(len(report['rows']) == 1600 and actual_keys == expected_keys
                and len(report['folds']) == 100 and all(f['complete'] for f in report['folds']),
                'All400 rows per sensitivity and all100 participant/fold/sensitivity units required')
        report['all_1600_rows_recorded'] = True
        report['complete_primary_sensitivities'] = all(r['status'] == 'evaluable' for r in report['rows'] if r['roi_role'] == 'primary')
        report['EVC_evaluable_by_sensitivity'] = {s: all(r['status'] == 'evaluable' for r in report['rows']
            if r['sensitivity'] == s and r['roi_role'] == 'diagnostic') for s in SENSITIVITIES}
        report['descriptive_effects'] = {s: panel_effects(np, native, report['rows'], s) for s in SENSITIVITIES}
        check()
        report['complete_calculation'] = True
        report['status'] = 'native_sensitivities_completed' if report['complete_primary_sensitivities'] else 'native_sensitivities_not_evaluable'
    except Exception as exc:
        incomplete(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            incomplete(TimeoutError('Deadline reached during finalization'))
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_calculation'] and time.monotonic() >= deadline:
            incomplete(TimeoutError('Deadline reached during final publication'))
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_calculation'] and report['complete_primary_sensitivities'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
