#!/usr/bin/env python3
"""Admit the complete EP04 panel and publish its one mathematical decision.

Consumes named completed score reports, saved score arrays and structural
identity/context records only. No raw images, betas, embeddings or voxel arrays
are read. Admission failure never creates the fixed decision artifact. A
selected row still requires the registered exact rerun before any lock.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import time

import collect_identical_fold_encoding as enc
import collect_registered_nulls as nulls
import run_native_sensitivities as sens


panel = nulls.panel
unit = nulls.unit
EPISODE = panel.EPISODE
OBSERVED = unit.OBSERVED
SENSITIVITIES = EPISODE/'outputs/runs/native_sensitivities/20261005-all01/native_sensitivities.json'
DECISION = EPISODE/'outputs/runs/panel_decision/registered_panel_decision.json'
CONTEXT_RSA = {
    'sub-01': EPISODE/'outputs/runs/training_context_rsa/20261005-rsa-pilot-46658777/sub-01/training_context_rsa.json',
    **{person: EPISODE/'outputs/runs/training_context_rsa/20261005-rsa-remaining-46660232'/person/'training_context_rsa.json'
       for person in ('sub-03', 'sub-05', 'sub-06', 'sub-07')},
}
MAX_SECONDS = 600
require = panel.require


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def same_numeric(np, actual, expected, message):
    left, right = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
            and np.allclose(left, right, rtol=0, atol=unit.ENGINEERING_ATOL), message)


def completed_clock(record, observed, end, first_key, end_key):
    require(record[first_key] == observed['first_panel_evaluation_utc']
            and datetime.fromisoformat(record[end_key]) == datetime.fromisoformat(end),
            'Input changed the original panel clock')
    start, finish = (datetime.fromisoformat(record[key]) for key in ('started_utc', 'finished_utc'))
    require(start.tzinfo is not None and finish.tzinfo is not None
            and datetime.fromisoformat(observed['finished_utc']) <= start <= finish
            <= datetime.fromisoformat(end) and finish <= datetime.now(timezone.utc),
            'Input is not a completed calculation inside the original panel clock')


def metadata_inputs(np, observed, observed_rows, check):
    """Reuse the producers' exact metadata checks without opening source arrays."""
    metadata = panel.saved.json_lines(panel.METADATA/'development_images.jsonl')
    expected = {row['image_id']: row for row in metadata}
    require(len(metadata) == len(expected) == 23560
            and all(sum(row['participant'] == p for row in metadata) == 4712 for p in panel.PARTICIPANTS)
            and all(type(row['fold']) is int and row['fold'] in range(5)
                    and row['source_stratum'] in panel.STRATA for row in metadata),
            'Frozen development image/fold/source coverage differs')
    embedded = panel.saved.json_lines(panel.saved.EMBEDDINGS/'development_rows.jsonl')
    embedding_index = {row['image_id']: row for row in embedded}
    require(len(embedded) == len(embedding_index) == 23560 and set(embedding_index) == set(expected)
            and all(row['output_row'] == index and all(row[key] == expected[row['image_id']][key]
                    for key in ('participant', 'fold', 'source_stratum', 'component_id'))
                    for index, row in enumerate(embedded)), 'Embedding identity mapping differs')
    low_ids = panel.read_json(unit.LOW_LEVEL/'image_ids.json')
    require(low_ids == sorted(expected), 'Low-level full image identity list differs')
    low_index = {identifier: index for index, identifier in enumerate(low_ids)}
    inputs = {person: panel.participant_inputs(np, person, expected, check) for person in panel.PARTICIPANTS}
    base = dict(observed, _report_path=str(OBSERVED))
    folds = {(row['participant'], row['held_fold']): row for row in observed['folds']}
    sens.bind_base_support(base, observed_rows, folds, inputs, embedding_index, low_index, check)
    return expected, embedding_index, low_index, inputs


def admit_nulls(np, native, path, observed, rows, values, partial, wall_end, check):
    require(path.name == nulls.REPORT_NAME and path.is_relative_to(EPISODE/'outputs/runs/registered_null_collection'),
            'Null report path must identify an allowed EP04 collection before reading')
    record = panel.read_json(path)
    require(record.get('analysis_id') == 'controls_diagnostics'
            and record.get('mode') == 'complete_registered_null_collection'
            and record.get('status') == 'complete_registered_nulls_without_candidate_decision'
            and all(record.get(key) is True for key in ('complete_registered_nulls',
                'complete_declared_coverage', 'complete_score_coverage', 'signed_null_statistics_preserved'))
            and all(record.get(key) is False for key in ('candidate_screen_performed',
                'candidate_selection_performed', 'candidate_lock_authorized', 'raw_source_or_feature_arrays_opened'))
            and record.get('errors') == [] and Path(record['observed_report']).resolve() == OBSERVED,
            'Only the complete registered-null collector is admissible')
    completed_clock(record, observed, wall_end, 'first_panel_evaluation_utc', 'panel_wall_deadline_utc')
    require(record['registered_permutation_draws'] == 10000 and record['registered_geometry_draws_per_VLM'] == 1000
            and record['model_order'] == list(unit.MODEL_ORDER) and record['VLM_order'] == list(unit.VLM_ORDER)
            and record['participant_order'] == list(panel.PARTICIPANTS) and record['fold_order'] == list(range(5))
            and record['ROI_order'] == list(unit.ROI_ORDER) and record['quantile'] == .95
            and record['quantile_method'] == 'higher' and record['comparison_rule'] == 'strictly_greater_than'
            and record['positive_observed_alignment_required_separately'] is True
            and record['complete_score_arrays'] == 'complete_null_scores.npz', 'Null count/order/statistic contract differs')
    require(record['unit_reports'] and len(record['unit_reports']) == len(set(record['unit_reports'])),
            'Null collector lacks its explicit unique completed-unit list')
    identities = [native.ModelIdentity(panel.MODELS[m][1], panel.MODELS[m][0]) for m in panel.VLM_LABELS]
    vlm_indices = [unit.MODEL_ORDER.index(m) for m in panel.VLM_LABELS]
    dino_index = unit.MODEL_ORDER.index('DINOv2')
    # These are completed null-score arrays, never source embeddings or neural patterns.
    score_path = (path.parent/'complete_null_scores.npz').resolve()
    require(score_path.parent == path.parent, 'Completed null-score arrays cannot redirect outside their collection')
    with np.load(score_path, allow_pickle=False) as arrays:
        same_numeric(np, arrays['observed_component_z'], values, 'Null collector observed native scores differ')
        same_numeric(np, arrays['observed_partial_component_z'], partial, 'Null collector partial scores differ')
        permutations, geometries = arrays['permutation_component_z'], arrays['geometry_component_z']
        require(permutations.shape == (10000, 4, 5, 5, 3) and geometries.shape == (1000, 3, 5, 5, 3)
                and np.isfinite(permutations).all() and np.isfinite(geometries).all(),
                'Every registered null draw and all25 cells are required')
        thresholds = []
        for family, random_vlm, dino, threshold_key in (
            ('permutation', permutations[:, vlm_indices], permutations[:, dino_index], 'image_permutation_threshold'),
            ('geometry', geometries, values[dino_index], 'random_geometry_threshold')):
            chosen, statistics, threshold = nulls.familywise_distribution(np, native, random_vlm, dino, identities, check)
            require(np.array_equal(chosen, arrays[f'{family}_selected_VLM_index']), 'Null unconditional selection differs')
            same_numeric(np, arrays[f'{family}_signed_statistic'], statistics, 'Null signed statistic differs')
            require(finite(record[threshold_key]) and record[threshold_key] == threshold, 'Frozen higher95% threshold differs')
            thresholds.append(threshold)
    observed_effects = native.panel_effects(values[vlm_indices], values[dino_index])
    partial_effects = native.panel_effects(partial[vlm_indices], partial[dino_index])
    declared = record['all_three_observed_screen_inputs']
    require(len(declared) == 3, 'All three null-screen input rows required')
    for index, (entry, model) in enumerate(zip(declared, panel.VLM_LABELS)):
        require(entry['model_label'] == model and entry['model_id'] == panel.MODELS[model][1]
                and entry['dimension'] == panel.MODELS[model][0], 'Null observed model binding differs')
        for key, effects in (('native', observed_effects), ('low_level_partial', partial_effects)):
            for name, numbers in effects.items():
                same_numeric(np, entry[key][name], numbers[index], 'Null observed effect summary differs')
    return thresholds


def admit_sensitivities(np, native, observed, observed_rows, expected, inputs, wall_end, check):
    record = panel.read_json(SENSITIVITIES)
    require(record.get('analysis_id') == 'controls_diagnostics'
            and record.get('mode') == 'registered_observed_native_sensitivities'
            and record.get('status') == 'native_sensitivities_completed'
            and all(record.get(key) is True for key in ('all_1600_rows_recorded', 'complete_primary_sensitivities',
                'complete_calculation', 'base400_support_matched_before_scores'))
            and all(record.get(key) is False for key in ('candidate_screen_performed', 'candidate_selection_performed',
                'candidate_lock_authorized', 'original_panel_or_partial_RSA_recomputed', 'preprocessing_or_NC_refitted'))
            and record.get('errors') == [] and Path(record['observed_report']).resolve() == OBSERVED
            and record['models'] == observed['models'] and record['sensitivity_order'] == list(sens.SENSITIVITIES),
            'Complete unchanged registered sensitivities required')
    completed_clock(record, observed, wall_end, 'original_first_panel_evaluation_utc', 'original_panel_deadline_utc')
    representatives = sens.frozen_representatives(expected)
    require(Path(record['global_representative_file']).resolve() == SENSITIVITIES.parent/'global_component_representatives.json'
            and panel.read_json(Path(record['global_representative_file'])) == representatives,
            'Outcome-independent global near-neighbour representatives differ')
    rows = record['rows']
    keyed = {(r['sensitivity'], r['participant'], r['held_fold'], r['roi'], r['model_label']): r for r in rows}
    keys = {(s, p, f, r, m) for s in sens.SENSITIVITIES for p in panel.PARTICIPANTS
            for f in range(5) for r in panel.ROIS for m in panel.MODELS}
    require(len(rows) == len(keyed) == 1600 and set(keyed) == keys, 'All1600 unique sensitivity rows required')
    folds = {(r['sensitivity'], r['participant'], r['held_fold']): r for r in record['folds']}
    require(len(record['folds']) == len(folds) == 100 and set(folds) == {
        (s, p, f) for s in sens.SENSITIVITIES for p in panel.PARTICIPANTS for f in range(5)}
        and all(row.get('complete') is True for row in folds.values()), 'All100 complete sensitivity cells required')
    for person, (images, base_folds) in inputs.items():
        for fold in base_folds:
            for name in sens.SENSITIVITIES:
                check()
                indices, counts, cells = sens.selection(np, name, images, fold, representatives)
                held = fold['held_fold']
                current = folds[(name, person, held)]
                identity = SENSITIVITIES.parent/f'{person}-fold-{held}-{name}-images.jsonl'
                require(Path(current['image_id_file']).resolve() == identity
                        and current['original_primary_images'] == len(fold['indices'])
                        and current['evaluation_images'] == len(indices) and current['source_counts'] == counts,
                        'Sensitivity changed original subset/source support')
                admitted = panel.saved.json_lines(identity)
                require(len(admitted) == len(indices) and all(
                    r['image_id'] == images[int(i)]['image_id'] and r['pattern_row'] == int(i)
                    and r['source_stratum'] == images[int(i)]['source_stratum']
                    and r['component_id'] == images[int(i)]['component_id']
                    and r['primary_valid_repeats'] == fold['validity'][int(i)]['primary_valid_repeats']
                    for r, i in zip(admitted, indices)), 'Sensitivity image/repeat identities differ')
                for roi in panel.ROIS:
                    for model in panel.MODELS:
                        row = keyed[(name, person, held, roi, model)]
                        base = observed_rows[(person, held, roi, model)]
                        require(row['model_id'] == base['model_id'] and row['roi_role'] == base['roi_role']
                                and row['retained_voxels'] == base['retained_voxels']
                                and row['training_context_common_images'] == base['training_context_common_images']
                                and row['training_repeat_reliability'] == base['training_repeat_reliability']
                                and row['training_provider_formula_NC_percent'] == base['training_provider_formula_NC_percent']
                                and row['evaluation_images'] == len(indices) and row['source_counts'] == counts
                                and Path(row['evaluation_image_id_file']).resolve() == identity
                                and row['neural_distance'] == ('cosine' if name == 'neural_cosine' else 'correlation'),
                                'Sensitivity model/ROI/support/context binding differs')
                        if roi not in panel.ROIS[:3]:
                            require(row['status'] in ('evaluable', 'not_evaluable'), 'Diagnostic sensitivity has no explicit status')
                            continue  # EVC sensitivity is descriptive, not a new required pass gate.
                        require(row['status'] == 'evaluable' and finite(row['fisher_z']), 'Required primary sensitivity is not evaluable')
                        components = row['source_cell_results']
                        require(len(components) == len(cells), 'Sensitivity source-cell coverage differs')
                        zs = []
                        for component, (stratum, cell_indices) in zip(components, cells):
                            require(component['source_cell'] == stratum and component['evaluation_images'] == len(cell_indices)
                                    and component['status'] == 'evaluable', 'Missing sensitivity source cell')
                            zs.append(float(nulls.scores(np, native, component, ())[()]))
                        require(abs(row['fisher_z']-float(np.mean(zs))) <= unit.ENGINEERING_ATOL,
                                'Sensitivity equal-source Fisher aggregation differs')
    arrays = {}
    for name in sens.SENSITIVITIES:
        arrays[name] = np.array([[[[keyed[(name, p, f, r, m)]['fisher_z'] for r in panel.ROIS[:3]]
            for f in range(5)] for p in panel.PARTICIPANTS] for m in panel.MODELS])
    return arrays


def admit_encoding(np, native, path, observed, observed_rows, inputs, embedding_index, low_index, wall_end, check):
    require(path.name == enc.REPORT_NAME and path.is_relative_to(EPISODE/'outputs/runs/identical_fold_encoding_collection'),
            'Encoding report path must identify an allowed EP04 collection before reading')
    record = panel.read_json(path)
    require(record.get('analysis_id') == 'controls_diagnostics'
            and record.get('diagnostic') == 'identical_fold_encoding_collection'
            and record.get('status') == 'encoding_diagnostic_complete'
            and all(record.get(key) is True for key in ('all_25_units_recorded', 'all_400_rows_recorded',
                'all_support_bindings_matched', 'complete_primary_encoding', 'complete_EVC_encoding', 'complete_encoding_diagnostic'))
            and record.get('diagnostics', {}).get('identical_fold_encoding') is True
            and all(record.get(key) is False for key in ('candidate_screen_performed', 'candidate_selection_performed',
                'candidate_lock_authorized', 'encoding_sign_used_as_gate', 'pooled_SSE_across_folds_or_participants',
                'Fisher_transform_of_R2', 'EVC_in_primary_aggregation', 'raw_neural_feature_or_voxel_arrays_reopened'))
            and record.get('errors') == [] and Path(record['observed_report']).resolve() == OBSERVED,
            'Complete identical-fold encoding diagnostic required')
    completed_clock(record, observed, wall_end, 'first_panel_evaluation_utc', 'panel_clock_deadline_utc')
    paths = [Path(value).resolve() for value in record['unit_reports']]
    require(len(paths) == len(set(paths)) == 25 and all(path.name == enc.encoding.REPORT_NAME
            and path.is_relative_to(EPISODE/'outputs/runs/identical_fold_encoding') for path in paths),
            'All25 explicit encoding units required')
    units = enc.unit_map([(p, panel.read_json(p)) for p in paths])
    rows = []
    for person, (images, folds) in inputs.items():
        for fold in folds:
            check()
            source, producer = units[(person, fold['held_fold'])]
            enc.validate_header(source, producer, observed, wall_end)
            rows.extend(enc.bind_cell(source, producer, OBSERVED, observed, observed_rows,
                                      images, fold, embedding_index, low_index))
    key = lambda row: (row['participant'], row['held_fold'], row['roi'], row['model_label'])
    require(sorted(rows, key=key) == sorted(record['rows'], key=key), 'Encoding collection rows differ from admitted units')
    primary, evc, effects, evc_summary = enc.summarize(np, native, rows)
    require(primary and evc and effects == record['primary_encoding_effects']
            and evc_summary == record['EVC_encoding_summary'], 'Encoding aggregation/status differs')
    comparison, raw = enc.observed_diagnostic_tables(np, native, observed_rows, rows, effects)
    require(comparison == record['native_RSA_vs_encoding'] and raw == record['raw_high_level_vs_EVC'],
            'Concrete RSA/encoding or raw EVC/context diagnostic table differs')
    return {'native_RSA_vs_encoding': comparison, 'raw_high_level_vs_EVC': raw,
            'primary_encoding_effects': effects, 'EVC_encoding_summary': evc_summary}


def admit_matched_context(np, native, observed, inputs, embedding_index, check):
    rows = []
    for person, (images, folds) in inputs.items():
        check()
        path = CONTEXT_RSA[person]
        record = panel.read_json(path)
        require(record.get('analysis_id') == 'training_neural_context' and record.get('mode') == 'matched_training_context_RSA'
                and record.get('participant') == person and record.get('status') == 'participant_training_context_RSA_completed'
                and record.get('complete_participant_training_context_RSA') is True
                and record.get('all_required_rows_recorded') is True and record.get('errors') == []
                and all(record.get(key) is False for key in ('primary_panel_evaluated', 'candidate_selection_performed',
                    'held_out_performance', 'raw_beta_arrays_read', 'raw_images_or_excluded_outcomes_opened',
                    'preprocessing_or_noise_ceiling_refitted'))
                and record.get('null_draws') == 0 and record['models'] == observed['models']
                and Path(record['preprocessing_report']).resolve() == panel.saved.preprocessing_report(person).resolve()
                and Path(record['context_report']).resolve() == panel.saved.CONTEXT_REPORTS[person].resolve(),
                'Complete frozen matched-training RSA evidence required')
        require(datetime.fromisoformat(record['finished_utc']) <= datetime.fromisoformat(observed['first_panel_evaluation_utc']),
                'Training context was not frozen before primary scoring')
        keyed = {(r['held_fold'], r['roi'], r['model_label']): r for r in record['rows']}
        require(len(record['rows']) == len(keyed) == 80 and set(keyed) == {
            (f, r, m) for f in range(5) for r in panel.ROIS for m in panel.MODELS}, 'All80 matched-training rows required per participant')
        recorded_folds = {r['held_fold']: r for r in record['folds']}
        require(len(record['folds']) == len(recorded_folds) == 5 and set(recorded_folds) == set(range(5)),
                'Every matched-training fold required')
        for fold in folds:
            held = fold['held_fold']
            indices = [i for i, (image, valid) in enumerate(zip(images, fold['validity'])) if image['fold'] != held
                       and valid['primary_eligible'] and valid['evc_eligible_on_primary_images']]
            identity = path.parent/f'fold-{held}-training_images.jsonl'
            required_ids = [{'image_id': images[i]['image_id'], 'pattern_row': i,
                             'embedding_row': embedding_index[images[i]['image_id']]['output_row']} for i in indices]
            stored = recorded_folds[held]
            require(stored['complete'] is True and stored['matched_image_and_repeat_support'] is True
                    and stored['rows_recorded'] == 16 and stored['training_images'] == len(indices)
                    == fold['training']['common_training_images']
                    and Path(stored['training_image_id_file']).resolve() == identity
                    and panel.saved.json_lines(identity) == required_ids,
                    'Matched training-image support differs from frozen common support')
            for roi in panel.ROIS:
                context = fold['training']['rois'][roi]
                for model in panel.MODELS:
                    row = keyed[(held, roi, model)]
                    require(row['participant'] == person and row['model_id'] == panel.MODELS[model][1]
                            and row['training_images'] == len(indices) and row['retained_voxels'] == context['retained_voxels']
                            and Path(row['training_image_id_file']).resolve() == identity
                            and row['repeat_reliability'] == context['repeat_reliability']
                            and row['provider_formula_NC_percent'] == context['provider_formula_NC_percent']
                            and row['valid_training_presentations'] == context['valid_training_presentations']
                            and row['repeat_count_histogram'] == context['repeat_count_histogram']
                            and Path(row['context_per_voxel_artifact']).resolve() == Path(context['per_voxel_values']).resolve()
                            and row['status'] == 'evaluable', 'Matched training RSA/ROI/reliability binding differs')
                    nulls.scores(np, native, {'spearman_r': row['rsa_spearman_r'], 'fisher_z': row['rsa_fisher_z']}, ())
                    rows.append(dict(row, input_report=str(path)))
    require(len(rows) == 400, 'All400 matched-training context rows required')
    return rows


def screen_inputs(np, native, values, partial, sensitivities, thresholds, diagnostics):
    """Only arithmetic on already-admitted full five-fold panels; no I/O."""
    require(values.shape == partial.shape == (4, 5, 5, 3)
            and set(sensitivities) == set(sens.SENSITIVITIES)
            and all(array.shape == (4, 5, 5, 3) for array in sensitivities.values()),
            'Exactly four models, five people, five folds and three primary ROIs required')
    vi = [unit.MODEL_ORDER.index(m) for m in panel.VLM_LABELS]
    di = unit.MODEL_ORDER.index('DINOv2')
    partial_effects = native.panel_effects(partial[vi], partial[di])
    effects = {'low_level_partial_rsa': partial_effects,
               **{name: native.panel_effects(array[vi], array[di]) for name, array in sensitivities.items()}}
    statuses = {name: ['pass' if (result['gain'][i] > 0 and
                (name != 'low_level_partial_rsa' or np.all(result['participant_loo'][i] > 0))) else 'fail'
                for i in range(3)] for name, result in effects.items()}
    identities = [native.ModelIdentity(panel.MODELS[m][1], panel.MODELS[m][0]) for m in panel.VLM_LABELS]
    decision = native.screen_complete_panel(values[vi], values[di], identities, *thresholds, statuses, diagnostics)
    participant_rows, sector_rows = [], []
    for model_index, model in enumerate(panel.VLM_LABELS):
        difference = values[vi[model_index]]-values[di]
        for index, person in enumerate(panel.PARTICIPANTS):
            participant_rows.append({'model_label': model, 'model_id': panel.MODELS[model][1], 'participant': person,
                'candidate_mean_Fisher_z': float(values[vi[model_index], index].mean()),
                'DINOv2_mean_Fisher_z': float(values[di, index].mean()), 'gain_Fisher_z': float(difference[index].mean()),
                'participant_gain_positive': bool(difference[index].mean() > 0),
                'leave_this_participant_out_gain': decision['effects']['participant_loo'][model_index][index],
                'row_passes_complete_panel': all(decision['gates'][model_index].values())})
        for index, roi in enumerate(panel.ROIS[:3]):
            sector_rows.append({'model_label': model, 'roi': roi,
                'candidate_mean_Fisher_z': float(values[vi[model_index], :, :, index].mean()),
                'DINOv2_mean_Fisher_z': float(values[di, :, :, index].mean()),
                'gain_Fisher_z': float(difference[:, :, index].mean()),
                'leave_this_ROI_out_gain': decision['effects']['sector_loo'][model_index][index]})
    return {**decision, 'falsifier_statuses': statuses,
            'sensitivity_effects': {name: {key: value.tolist() for key, value in result.items()} for name, result in effects.items()},
            'all_participant_effects': participant_rows, 'all_primary_ROI_effects': sector_rows}


def publish_decision(path, report, check):
    """One exclusive, atomic publication after complete admission; never overwrite."""
    require(report.get('complete_admission') is True and report.get('candidate_decisions') == 1,
            'Admission failure cannot publish a scientific candidate decision')
    payload = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    check()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    created = False
    try:
        with temporary.open('xb') as stream:
            created = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        check()
        os.link(temporary, path)  # Fails if the single decision path already exists.
    finally:
        if created:
            temporary.unlink()


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--null-report', required=True, type=Path)
    parser.add_argument('--encoding-report', required=True, type=Path)
    parser.add_argument('--max-seconds', required=True, type=int)
    args = parser.parse_args(argv)
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in socket.gethostname()
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(name) == '1' for name in panel.saved.THREAD_ENV),
            'One actual compute CPU and all six numerical thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS, 'Use EP04 cwd and process cap<=600s')
    require(not DECISION.exists(), 'The one fixed candidate-decision artifact already exists; reconcile it without replay')
    deadline = started+args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Screen admission or original120-hour panel deadline reached')

    def interrupted(*_):
        raise TimeoutError('Screen admission interrupted')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        observed = panel.read_json(OBSERVED)
        observed_rows = unit.validate_observed(observed, OBSERVED)
        deadline, wall_end = unit.clock_deadline(observed['first_panel_evaluation_utc'], datetime.now(timezone.utc),
                                                time.monotonic(), deadline)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        import numpy as np
        import native_rsa as native
        values, partial = nulls.observed_values(np, native, observed, observed_rows)
        for row in observed_rows.values():
            require(row['native_status'] == 'evaluable', 'Required raw native/EVC row is unavailable')
            nulls.scores(np, native, {'spearman_r': row['native_spearman_r'], 'fisher_z': row['native_fisher_z']}, ())
        expected, embedding_index, low_index, inputs = metadata_inputs(np, observed, observed_rows, check)
        thresholds = admit_nulls(np, native, args.null_report.resolve(), observed, observed_rows, values, partial, wall_end, check)
        sensitivity_values = admit_sensitivities(np, native, observed, observed_rows, expected, inputs, wall_end, check)
        diagnostics = admit_encoding(np, native, args.encoding_report.resolve(), observed, observed_rows, inputs,
                                     embedding_index, low_index, wall_end, check)
        matched = admit_matched_context(np, native, observed, inputs, embedding_index, check)
        check()
        decision = screen_inputs(np, native, values, partial, sensitivity_values, thresholds,
                                 {'identical_fold_encoding': True, 'evc_reliability_context': True})
        selected = decision['selected_index']
        report = {'analysis_id': 'candidate_decision', 'mode': 'complete_registered_fixed_panel_screen',
            'complete_admission': True, 'candidate_decisions': 1, 'candidate_selection_performed': True,
            'status': 'closed_no_candidate' if selected is None else 'selected_candidate_pending_exact_full_rerun',
            'selected_model': None if selected is None else {'model_label': panel.VLM_LABELS[selected],
                'model_id': panel.MODELS[panel.VLM_LABELS[selected]][1], 'dimension': panel.MODELS[panel.VLM_LABELS[selected]][0]},
            'VLM_order': list(panel.VLM_LABELS), 'participant_order': list(panel.PARTICIPANTS),
            'fold_order': list(range(5)), 'primary_ROI_order': list(panel.ROIS[:3]),
            'observed_report': str(OBSERVED), 'null_report': str(args.null_report.resolve()),
            'sensitivity_report': str(SENSITIVITIES), 'encoding_report': str(args.encoding_report.resolve()),
            'matched_training_context_reports': {person: str(path) for person, path in CONTEXT_RSA.items()},
            'first_panel_evaluation_utc': observed['first_panel_evaluation_utc'], 'panel_wall_deadline_utc': wall_end,
            'image_permutation_threshold': thresholds[0], 'random_geometry_threshold': thresholds[1],
            'null_quantile_method': 'higher', 'null_exceedance': 'strictly_greater_than',
            'diagnostics': {'identical_fold_encoding': True, 'evc_reliability_context': True},
            'diagnostic_tables': {**diagnostics, 'matched_training_RSA_and_reliability_context': matched},
            **decision, 'exact_full_development_rerun_performed': False, 'candidate_lock_authorized': False,
            'OOD_access_authorized': False, 'OOD_outcomes_opened': False,
            'raw_neural_image_feature_or_voxel_arrays_opened': False, 'automatic_retry': False,
            'automatic_successor': False, 'directory_discovery': False,
            'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
            'interpretation': 'Selection-exposed development decision; diagnostics cannot rescue primary RSA; exact rerun and lock remain separate'}
        publish_decision(DECISION, report, check)
        print(json.dumps({'decision_file': str(DECISION), 'status': report['status'], 'selected_model': report['selected_model']}), flush=True)
    except Exception as exc:
        print(json.dumps({'screen_admission_failed': not DECISION.exists(), 'decision_path_exists': DECISION.exists(),
                          'type': type(exc).__name__, 'message': str(exc)}), flush=True)
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
