#!/usr/bin/env python3
"""Collect all25 frozen encoding units and retain all400 diagnostic rows.

Only explicit unit reports and existing structural/image-identity records are
read. Neural, feature and voxel arrays are not reopened. Producer-verified voxel
identity equality is bound through the same preprocessing/context and counts.
Negative R2 and explicit unavailable EVC rows remain visible; no candidate is
screened, selected, rescued or locked by this diagnostic.
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

import run_identical_fold_encoding as encoding
import run_registered_null_unit as null_unit


panel = encoding.observed
EPISODE = panel.EPISODE
REPORT_NAME = 'identical_fold_encoding_collection.json'
MAX_SECONDS = 600
require = panel.require


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_map(named_reports):
    result = {}
    for path, report in named_reports:
        key = (report['participant'], report['held_fold'])
        require(key[0] in panel.PARTICIPANTS and type(key[1]) is int and key[1] in range(5)
                and key not in result, 'Duplicate or invalid encoding participant/fold unit')
        result[key] = (path, report)
    require(set(result) == {(person, fold) for person in panel.PARTICIPANTS for fold in range(5)},
            'Exactly25 participant/fold encoding units required; no incomplete-subset diagnostic')
    return result


def checked_score(row):
    """Completeness and evaluability differ; undefined scores are never zero."""
    require(row.get('complete') is True and row.get('status') in ('evaluable', 'not_evaluable'),
            'Every encoding row must have a terminal explicit status')
    require(type(row['processed_voxels']) is int and 0 <= row['processed_voxels'] <= row['retained_voxels']
            and finite(row['SSE_partial']) and row['SSE_partial'] >= 0
            and finite(row['SSE_baseline_partial']) and row['SSE_baseline_partial'] >= 0,
            'Invalid processed support or partial squared-error accounting')
    if row['status'] == 'not_evaluable':
        require(row['R2_SSE'] is None and isinstance(row.get('reason'), str) and row['reason'],
                'Unavailable row needs a reason and a null score')
        return None
    require(row['processed_voxels'] == row['retained_voxels']
            and finite(row['SSE']) and row['SSE'] >= 0
            and finite(row['SSE_baseline']) and row['SSE_baseline'] > 0
            and row['SSE'] == row['SSE_partial']
            and row['SSE_baseline'] == row['SSE_baseline_partial'],
            'Evaluable score requires every voxel and a positive training-mean baseline')
    expected = encoding.finish_score(row['SSE'], row['SSE_baseline'])
    require(finite(row['R2_SSE']) and expected is not None
            and math.isclose(row['R2_SSE'], expected, rel_tol=1e-12, abs_tol=1e-12),
            'R2 differs from the unmodified full-matrix SSE/training-mean baseline')
    return row['R2_SSE']


def direction_comparison(rsa_gain, encoding_gain):
    if rsa_gain is None or encoding_gain is None:
        return 'not_evaluable'
    if rsa_gain == 0 or encoding_gain == 0:
        return 'at_least_one_exact_zero'
    if rsa_gain > 0 and encoding_gain > 0:
        return 'same_positive_direction'
    if rsa_gain < 0 and encoding_gain < 0:
        return 'same_negative_direction'
    return 'opposite_directions_readout_dependence'


def summarize(np, native, rows):
    require(len(rows) == 400, 'All400 encoding rows must remain present')
    keyed = {(row['participant'], row['held_fold'], row['roi'], row['model_label']): row for row in rows}
    expected = {(p, f, r, m) for p in panel.PARTICIPANTS for f in range(5)
                for r in panel.ROIS for m in panel.MODELS}
    require(len(keyed) == 400 and set(keyed) == expected, 'Missing or duplicate encoding row identity')
    primary_complete = all(row['status'] == 'evaluable' for row in rows if row['roi'] in panel.ROIS[:3])
    evc_complete = all(row['status'] == 'evaluable' for row in rows if row['roi'] == panel.ROIS[3])
    primary_effects = None
    if primary_complete:
        def values(model):
            return np.array([[[keyed[(p, f, r, model)]['R2_SSE'] for r in panel.ROIS[:3]]
                              for f in range(5)] for p in panel.PARTICIPANTS])
        # This helper is equal-weight contrast arithmetic. R2 is not Fisher transformed.
        effects = native.panel_effects(np.array([values(model) for model in panel.VLM_LABELS]), values('DINOv2'))
        primary_effects = {key: value.tolist() for key, value in effects.items()
                           if key != 'unsubtracted_alignment'}
        primary_effects['mean_VLM_R2'] = effects['unsubtracted_alignment'].tolist()
        primary_effects['mean_DINOv2_R2'] = float(values('DINOv2').mean())
    evc = []
    for model in panel.MODELS:
        selected = [keyed[(p, f, panel.ROIS[3], model)] for p in panel.PARTICIPANTS for f in range(5)]
        complete = all(row['status'] == 'evaluable' for row in selected)
        values = (np.array([row['R2_SSE'] for row in selected]).reshape(5, 5) if complete else None)
        evc.append({'model_label': model, 'model_id': panel.MODELS[model][1],
                    'status': 'evaluable' if complete else 'not_evaluable',
                    'participant_mean_R2': values.mean(axis=1).tolist() if complete else None,
                    'equal_participant_mean_R2': float(values.mean()) if complete else None,
                    'unavailable_cells': [{'participant': row['participant'], 'held_fold': row['held_fold'],
                                           'reason': row['reason']} for row in selected if row['status'] != 'evaluable']})
    return primary_complete, evc_complete, primary_effects, evc


def validate_header(path, report, observed, wall_end):
    person = report['participant']
    require(report.get('analysis_id') == 'controls_diagnostics'
            and report.get('diagnostic') == 'identical_fold_encoding'
            and report.get('status') in ('encoding_diagnostic_complete', 'encoding_diagnostic_not_evaluable')
            and all(report.get(key) is True for key in (
                'all_16_rows_recorded', 'full_feature_coverage_qualified', 'observed_support_matched'))
            and all(report.get(key) is False for key in (
                'raw_beta_arrays_read', 'raw_images_or_excluded_outcomes_opened',
                'preprocessing_or_NC_refitted', 'candidate_selection_performed', 'candidate_lock_authorized',
                'lambda_or_rank_search_performed', 'automatic_retry', 'automatic_resume'))
            and report.get('errors') == [], f'Encoding unit is not a complete terminal calculation: {path}')
    require(report['normalized_loss_lambda'] == encoding.LAMBDA
            and report['intercept'] == 'unpenalized_training_mean'
            and report['model_input'] == 'row_L2_native_embedding'
            and report['evaluation_support'] == 'exact_primary_RSA_IDs_without_EVC_pruning'
            and Path(report['observed_report']).resolve() == null_unit.OBSERVED
            and report['observed_job_id'] == observed['job_id']
            and Path(report['low_level_report']).resolve() == null_unit.LOW_LEVEL/'low_level_preparation_report.json'
            and Path(report['preprocessing_report']).resolve() == panel.saved.preprocessing_report(person).resolve()
            and Path(report['training_context_report']).resolve() == panel.saved.CONTEXT_REPORTS[person].resolve(),
            'Fixed ridge, observed or preprocessing/context binding differs')
    first, start, finish, end = (datetime.fromisoformat(report[key]) for key in (
        'first_panel_evaluation_utc', 'started_utc', 'finished_utc', 'panel_clock_deadline_utc'))
    require(all(value.tzinfo is not None for value in (first, start, finish, end))
            and first == datetime.fromisoformat(observed['first_panel_evaluation_utc'])
            and end == datetime.fromisoformat(wall_end)
            and datetime.fromisoformat(observed['finished_utc']) <= start <= finish <= end
            and finish <= datetime.now(timezone.utc), 'Encoding unit changed or exceeded the original panel clock')


def bind_cell(path, report, observed_path, observed, observed_rows, images, fold,
              embedding_index, low_index):
    person, held = report['participant'], report['held_fold']
    identity, _ = encoding.observed_support(observed_path, observed, null_unit.LOW_LEVEL,
                                            person, fold, images, embedding_index, low_index)
    require(Path(report['evaluation_image_id_file']).resolve() == identity, 'Encoding evaluation identity file differs')
    training_path = (path.parent/'training_images.jsonl').resolve()
    require(training_path.parent == path.parent and Path(report['training_image_id_file']).resolve() == training_path,
            'Encoding training identity file differs')
    primary_train, evaluation, _ = encoding.support_indices(images, fold['validity'], held)
    evc_train, evc_evaluation, missing = encoding.support_indices(images, fold['validity'], held, evc=True)
    require(evaluation == evc_evaluation == fold['indices'].tolist(), 'Encoding must retain the complete primary evaluation support')
    evc_set = set(evc_train)
    expected_training = [dict(image_id=images[i]['image_id'], pattern_row=i,
        embedding_row=embedding_index[images[i]['image_id']]['output_row'], fold=images[i]['fold'],
        component_id=images[i]['component_id'], source_stratum=images[i]['source_stratum'],
        primary_valid_repeats=fold['validity'][i]['primary_valid_repeats'],
        evc_valid_repeats=fold['validity'][i]['evc_valid_repeats'], EVC_training_eligible=i in evc_set)
        for i in sorted(primary_train)]
    require(panel.saved.json_lines(training_path) == expected_training
            and not ({images[i]['component_id'] for i in primary_train}
                     & {images[i]['component_id'] for i in evaluation}),
            'Encoding training complement, eligibility or component separation differs')
    rows = report['rows']
    require(len(rows) == 16 and {(r['participant'], r['held_fold'], r['roi'], r['model_label']) for r in rows}
            == {(person, held, roi, model) for roi in panel.ROIS for model in panel.MODELS},
            'Exactly16 model/ROI rows required in every unit')
    for row in rows:
        roi, model = row['roi'], row['model_label']
        old = observed_rows[(person, held, roi, model)]
        trained = fold['training']['rois'][roi]
        evc = roi == panel.ROIS[3]
        require(row['model_id'] == old['model_id'] == panel.MODELS[model][1]
                and row['roi_role'] == old['roi_role'] == ('diagnostic' if evc else 'primary')
                and row['training_images'] == len(evc_train if evc else primary_train)
                and row['evaluation_images'] == old['evaluation_images'] == len(evaluation)
                and row['retained_voxels'] == old['retained_voxels'] == trained['retained_voxels']
                == fold['original']['roi_support'][roi]['retained_voxels']
                and Path(row['evaluation_image_id_file']).resolve() == identity
                and Path(row['training_image_id_file']).resolve() == training_path
                and row['EVC_missing_evaluation_image_ids'] == ([images[i]['image_id'] for i in missing] if evc else []),
                'Encoding model/image/voxel binding differs from the exact observed fold')
        require(old['training_context_common_images'] == fold['training']['common_training_images']
                and old['training_repeat_reliability'] == trained['repeat_reliability']
                and old['training_provider_formula_NC_percent'] == trained['provider_formula_NC_percent'],
                'Observed frozen training reliability/NC context differs')
        checked_score(row)
        if evc and missing:
            require(row['status'] == 'not_evaluable', 'EVC missingness cannot become an evaluable intersection')
    evaluable = all(row['status'] == 'evaluable' for row in rows)
    require(report['complete_encoding_diagnostic'] is evaluable
            and report['status'] == ('encoding_diagnostic_complete' if evaluable else 'encoding_diagnostic_not_evaluable'),
            'Unit completion flag conflicts with its sixteen explicit statuses')
    return [dict(row, input_report=str(path)) for row in rows]


def observed_diagnostic_tables(np, native, observed_rows, encoding_rows, encoding_effects):
    def primary_values(model):
        return np.array([[[observed_rows[(p, f, roi, model)]['native_fisher_z'] for roi in panel.ROIS[:3]]
                           for f in range(5)] for p in panel.PARTICIPANTS])
    rsa = native.panel_effects(np.array([primary_values(model) for model in panel.VLM_LABELS]), primary_values('DINOv2'))
    comparison = []
    for index, model in enumerate(panel.VLM_LABELS):
        gain = encoding_effects['gain'][index] if encoding_effects is not None else None
        participants = encoding_effects['participant'][index] if encoding_effects is not None else None
        comparison.append({'model_label': model, 'model_id': panel.MODELS[model][1],
            'native_RSA_gain_Fisher_z': float(rsa['gain'][index]), 'encoding_gain_R2': gain,
            'participant_native_RSA_gain_Fisher_z': rsa['participant'][index].tolist(),
            'participant_encoding_gain_R2': participants,
            'direction_comparison': direction_comparison(float(rsa['gain'][index]), gain),
            'participant_direction_comparison': [direction_comparison(float(rsa['participant'][index, p]),
                participants[p] if participants is not None else None) for p in range(5)],
            'interpretation': 'Descriptive readout direction only; no encoding sign is an eligibility or rescue gate'})
    raw = []
    encoding_keyed = {(r['participant'], r['held_fold'], r['roi'], r['model_label']): r for r in encoding_rows}
    for person in panel.PARTICIPANTS:
        for fold in range(5):
            for model in panel.MODELS:
                rois = {}
                for roi in panel.ROIS:
                    old = observed_rows[(person, fold, roi, model)]
                    encoded = encoding_keyed[(person, fold, roi, model)]
                    rois[roi] = {'native_spearman_r': old['native_spearman_r'],
                        'native_fisher_z': old['native_fisher_z'], 'retained_voxels': old['retained_voxels'],
                        'training_context_common_images': old['training_context_common_images'],
                        'training_repeat_reliability': old['training_repeat_reliability'],
                        'training_provider_formula_NC_percent': old['training_provider_formula_NC_percent'],
                        'encoding_status': encoded['status'], 'encoding_R2': encoded['R2_SSE']}
                first = observed_rows[(person, fold, panel.ROIS[0], model)]
                raw.append({'participant': person, 'held_fold': fold, 'model_label': model,
                    'evaluation_image_id_file': first['evaluation_image_id_file'],
                    'evaluation_images': first['evaluation_images'],
                    'training_context_report': str(panel.saved.CONTEXT_REPORTS[person]), 'rois': rois,
                    'equal_high_level_mean_z_minus_EVC_z': float(np.mean([
                        rois[roi]['native_fisher_z'] for roi in panel.ROIS[:3]])-rois[panel.ROIS[3]]['native_fisher_z']),
                    'context_status': 'Reliability/NC comes from the frozen common training images, not evaluation images',
                    'interpretation': 'Descriptive same-evaluation-image comparison; no specificity inference or candidate gate'})
    return comparison, raw


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--unit-reports', nargs='+', type=Path)
    inputs.add_argument('--report-list', type=Path, help='JSON array of explicit paths relative to EP04 cwd or absolute')
    parser.add_argument('--observed-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    host = socket.gethostname()
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in host and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(name) == '1' for name in panel.saved.THREAD_ENV),
            'One actual compute CPU and six numerical thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
            'Use EP04 cwd and an explicit process bound<=600s')
    output, observed_path = args.output_dir.resolve(), args.observed_report.resolve()
    require(not output.exists() and output.is_relative_to(EPISODE/'outputs/runs/identical_fold_encoding_collection')
            and observed_path == null_unit.OBSERVED, 'Fresh collector output and exact observed01 input required')
    output.mkdir(parents=True)
    deadline = started+args.max_seconds
    report = {'analysis_id': 'controls_diagnostics', 'diagnostic': 'identical_fold_encoding_collection',
        'status': 'started', 'all_25_units_recorded': False, 'all_400_rows_recorded': False,
        'all_support_bindings_matched': False, 'complete_primary_encoding': False, 'complete_EVC_encoding': False,
        'complete_encoding_diagnostic': False, 'diagnostics': {'identical_fold_encoding': False},
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': host, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'internal_seconds': args.max_seconds, 'observed_report': str(observed_path), 'unit_reports': [], 'rows': [], 'errors': [],
        'participant_order': list(panel.PARTICIPANTS), 'fold_order': list(range(5)), 'ROI_order': list(panel.ROIS),
        'model_order': list(panel.MODELS), 'VLM_order': list(panel.VLM_LABELS),
        'aggregation': 'Mean fold R2 and VLM-minus-DINOv2 differences; equal folds, primary sectors and participants',
        'pooled_SSE_across_folds_or_participants': False, 'Fisher_transform_of_R2': False,
        'negative_R2_preserved': True, 'EVC_in_primary_aggregation': False,
        'candidate_screen_performed': False, 'candidate_selection_performed': False, 'candidate_lock_authorized': False,
        'encoding_sign_used_as_gate': False, 'raw_neural_feature_or_voxel_arrays_reopened': False,
        'voxel_binding': 'Producer checks saved voxel-ID equality; collector binds unchanged preprocessing/context and counts',
        'evc_reliability_context_claimed_complete': False,
        'separate_matched_training_RSA_context_still_requires_its_existing_evidence': True,
        'automatic_retry': False, 'automatic_resume': False, 'directory_discovery': False}

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Encoding collection process or original120-hour deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output/'identical_fold_encoding_collection.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/REPORT_NAME)

    def fail(exc):
        report['complete_encoding_diagnostic'] = False
        report['diagnostics']['identical_fold_encoding'] = False
        report['status'] = 'technical_encoding_collection_failure'
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})

    def interrupted(*_):
        raise TimeoutError('Encoding collection interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        save()
        observed = panel.read_json(observed_path)
        observed_rows = null_unit.validate_observed(observed, observed_path)
        deadline, wall_end = null_unit.clock_deadline(observed['first_panel_evaluation_utc'], datetime.now(timezone.utc),
                                                   time.monotonic(), deadline)
        report['first_panel_evaluation_utc'] = observed['first_panel_evaluation_utc']
        report['panel_clock_deadline_utc'] = wall_end
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        paths = args.unit_reports if args.report_list is None else panel.read_json(args.report_list)
        require(isinstance(paths, list) and len(paths) == 25 and all(isinstance(p, (str, Path)) for p in paths),
                'Exactly25 explicit report paths required')
        paths = [Path(path).resolve() for path in paths]
        require(len(set(paths)) == 25 and all(path.name == encoding.REPORT_NAME and path.is_file()
                and path.is_relative_to(EPISODE/'outputs/runs/identical_fold_encoding') for path in paths),
                'Duplicate, missing or outside-scope encoding report')
        units = unit_map([(path, panel.read_json(path)) for path in paths])
        for path, record in units.values():
            validate_header(path, record, observed, wall_end)
        report['all_25_units_recorded'] = True
        report['unit_reports'] = [str(path) for path in paths]
        import numpy as np
        import native_rsa as native
        metadata = panel.saved.json_lines(panel.METADATA/'development_images.jsonl')
        expected = {row['image_id']: row for row in metadata}
        require(len(metadata) == len(expected) == 23560
                and all(sum(row['participant'] == p for row in metadata) == 4712 for p in panel.PARTICIPANTS),
                'Frozen development metadata coverage differs')
        embedding_rows = panel.saved.json_lines(panel.saved.EMBEDDINGS/'development_rows.jsonl')
        embedding_index = {row['image_id']: row for row in embedding_rows}
        require(len(embedding_rows) == len(embedding_index) == 23560 and set(embedding_index) == set(expected)
                and all(row['output_row'] == index for index, row in enumerate(embedding_rows)),
                'Frozen embedding row-identity coverage differs')
        low_index = {image_id: index for index, image_id in enumerate(sorted(expected))}
        for row in observed_rows.values():
            require(row['native_status'] == 'evaluable' and finite(row['native_spearman_r'])
                    and abs(row['native_spearman_r']) < 1 and finite(row['native_fisher_z'])
                    and math.isclose(native.fisher_z(row['native_spearman_r']), row['native_fisher_z'],
                                     rel_tol=1e-12, abs_tol=1e-12), 'Observed native row is nonfinite or inconsistent')
        for person in panel.PARTICIPANTS:
            check()
            images, folds = panel.participant_inputs(np, person, expected, check)
            for fold in folds:
                check()
                path, record = units[(person, fold['held_fold'])]
                report['rows'].extend(bind_cell(path, record, observed_path, observed, observed_rows,
                                                images, fold, embedding_index, low_index))
            save()
        report['all_support_bindings_matched'] = True
        primary, evc, effects, evc_summary = summarize(np, native, report['rows'])
        report['all_400_rows_recorded'] = True
        report['complete_primary_encoding'], report['complete_EVC_encoding'] = primary, evc
        report['primary_encoding_effects'] = effects
        report['EVC_encoding_summary'] = evc_summary
        comparison, raw = observed_diagnostic_tables(np, native, observed_rows, report['rows'], effects)
        report['native_RSA_vs_encoding'] = comparison
        report['raw_high_level_vs_EVC'] = raw
        check()
        report['complete_encoding_diagnostic'] = primary and evc
        report['diagnostics']['identical_fold_encoding'] = primary and evc
        report['status'] = 'encoding_diagnostic_complete' if primary and evc else 'encoding_diagnostic_not_evaluable'
    except Exception as exc:
        fail(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            fail(TimeoutError('Deadline reached during finalization'))
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_encoding_diagnostic'] and time.monotonic() >= deadline:
            fail(TimeoutError('Deadline reached during final publication'))
            save()
    return 0 if report['complete_encoding_diagnostic'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
