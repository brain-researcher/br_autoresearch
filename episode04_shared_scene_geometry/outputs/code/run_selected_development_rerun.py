#!/usr/bin/env python3
"""One conditional selected-row rerun, in 25 retained units and one collection.

The immutable decision supplies the candidate. No command-line model choice,
observed-candidate selection, configuration lock, audit access, or retry exists.
Nonselected null components stay immutable; every refreshed selected/DINO
component is compared before the original three-VLM null is reconstructed.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

import run_observed_native_panel as panel
import run_registered_null_unit as nullunit
import run_native_sensitivities as sens
import collect_registered_nulls as nullcollector


EPISODE = panel.EPISODE
DECISION = EPISODE / 'outputs/runs/panel_decision/registered_panel_decision.json'
OBSERVED = nullunit.OBSERVED
SENSITIVITIES = EPISODE / 'outputs/runs/native_sensitivities/20261005-all01/native_sensitivities.json'
NULL_REPORT = EPISODE / 'outputs/runs/registered_null_collection/20261005-full01/registered_null_collection.json'
NULL_SCORES = NULL_REPORT.parent / 'complete_null_scores.npz'
FAMILY = EPISODE / 'outputs/runs/selected_development_rerun/20261005-once01'
REPORT_NAME = 'selected_development_rerun.json'
PERMUTATIONS = 10000
GEOMETRIES = 1000
ATOL = nullunit.ENGINEERING_ATOL
require = panel.require
GATE_NAMES = {'practical_gain', 'participant_direction', 'participant_influence',
              'sector_influence', 'positive_alignment', 'image_null', 'geometry_null',
              'low_level_partial_rsa', *sens.SENSITIVITIES}


def fixed_json(path):
    require(not path.is_symlink() and path.resolve() == path, 'Fixed input must not redirect')
    return panel.read_json(path)


def coordinates(model):
    require(model in panel.VLM_LABELS, 'Candidate must be one frozen VLM')
    return panel.VLM_LABELS.index(model), list(panel.MODELS).index(model), list(panel.MODELS).index('DINOv2')


def slot_cell(slot):
    require(type(slot) is int and slot in range(25), 'Slot must identify one frozen participant/fold cell')
    return panel.PARTICIPANTS[slot // 5], slot % 5


def admit_decision(record):
    require(record.get('analysis_id') == 'candidate_decision'
            and record.get('mode') == 'complete_registered_fixed_panel_screen'
            and record.get('status') == 'selected_candidate_pending_exact_full_rerun'
            and record.get('complete_admission') is True
            and type(record.get('candidate_decisions')) is int and record['candidate_decisions'] == 1
            and record.get('candidate_selection_performed') is True,
            'Only the one admitted selected-candidate decision permits a rerun')
    require(all(record.get(key) is False for key in ('exact_full_development_rerun_performed',
                'candidate_lock_authorized', 'configuration_lock_authorized',
                'OOD_access_authorized', 'OOD_outcomes_opened')),
            'Rerun, configuration lock and OOD boundaries differ')
    selected = record['selected_index']
    require(type(selected) is int and selected in range(3)
            and record['VLM_order'] == list(panel.VLM_LABELS)
            and record['participant_order'] == list(panel.PARTICIPANTS)
            and record['fold_order'] == list(range(5))
            and record['primary_ROI_order'] == list(panel.ROIS[:3]), 'Frozen panel identity differs')
    model = panel.VLM_LABELS[selected]
    require(record['selected_model'] == {'model_label': model, 'model_id': panel.MODELS[model][1],
                                       'dimension': panel.MODELS[model][0]}, 'Selected model identity differs')
    require(len(record['gates']) == 3 and set(record['gates'][selected]) == GATE_NAMES
            and all(value is True for value in record['gates'][selected].values()),
            'The saved selected row must have passed every registered gate')
    for field, path in (('observed_report', OBSERVED), ('sensitivity_report', SENSITIVITIES),
                        ('null_report', NULL_REPORT)):
        require(Path(record[field]).resolve() == path, 'Decision source binding differs')
    require(record.get('diagnostics') == {'identical_fold_encoding': True, 'evc_reliability_context': True},
            'Existing required diagnostic completion must be retained')
    return model


def compare(np, actual, expected, name):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and actual.size > 0
            and np.isfinite(actual).all() and np.isfinite(expected).all(), name + ': shape/finite mismatch')
    difference = float(np.max(np.abs(actual - expected)))
    require(difference <= ATOL, name + ': replay differs beyond existing1e-12 engineering tolerance')
    return difference


def load_reference(decision, np, check):
    observed = fixed_json(OBSERVED)
    rows = nullunit.validate_observed(observed, OBSERVED)
    require(observed['first_panel_evaluation_utc'] == decision['first_panel_evaluation_utc'],
            'Original panel clock changed')
    sensitivity = fixed_json(SENSITIVITIES)
    require(sensitivity.get('status') == 'native_sensitivities_completed'
            and all(sensitivity.get(key) is True for key in ('all_1600_rows_recorded',
                'complete_primary_sensitivities', 'complete_calculation', 'base400_support_matched_before_scores'))
            and sensitivity.get('errors') == [] and Path(sensitivity['observed_report']).resolve() == OBSERVED,
            'Complete frozen sensitivity source required')
    sensitivity_rows = {(row['sensitivity'], row['participant'], row['held_fold'], row['roi'], row['model_label']): row
                        for row in sensitivity['rows']}
    require(len(sensitivity['rows']) == len(sensitivity_rows) == 1600,
            'All original sensitivity rows must be retained')
    nulls = fixed_json(NULL_REPORT)
    require(nulls.get('status') == 'complete_registered_nulls_without_candidate_decision'
            and all(nulls.get(key) is True for key in ('complete_registered_nulls',
                'complete_declared_coverage', 'complete_score_coverage'))
            and nulls.get('errors') == [] and Path(nulls['observed_report']).resolve() == OBSERVED
            and nulls['first_panel_evaluation_utc'] == decision['first_panel_evaluation_utc']
            and nulls['model_order'] == list(nullunit.MODEL_ORDER)
            and nulls['VLM_order'] == list(panel.VLM_LABELS)
            and nulls['participant_order'] == list(panel.PARTICIPANTS)
            and nulls['fold_order'] == list(range(5)) and nulls['ROI_order'] == list(panel.ROIS[:3])
            and nulls['complete_score_arrays'] == 'complete_null_scores.npz',
            'Complete frozen three-VLM null source required')
    for key in ('image_permutation_threshold', 'random_geometry_threshold'):
        require(nulls[key] == decision[key], 'Saved decision/null threshold differs')
    require(not NULL_SCORES.is_symlink() and NULL_SCORES.resolve() == NULL_SCORES,
            'Fixed null score input must not redirect')
    with np.load(NULL_SCORES, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    for key, shape in (('permutation_component_z', (PERMUTATIONS, 4, 5, 5, 3)),
                       ('geometry_component_z', (GEOMETRIES, 3, 5, 5, 3)),
                       ('observed_component_z', (4, 5, 5, 3)),
                       ('observed_partial_component_z', (4, 5, 5, 3))):
        require(arrays[key].shape == shape and np.isfinite(arrays[key]).all(),
                'Complete finite saved component array required: ' + key)
    check()
    return observed, rows, sensitivity_rows, arrays


def save_report(output, report):
    report['elapsed_seconds'] = time.monotonic() - report['_started']
    report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    saved = {key: value for key, value in report.items() if not key.startswith('_')}
    temporary = output / ('.' + REPORT_NAME + '.pending')
    with temporary.open('w') as stream:
        json.dump(saved, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / REPORT_NAME)


def primary_inputs(np, native, low_module, expected, embedding_index, features, low_index,
                   low_features, inputs, observed, rows, person, held, pair, check):
    base = dict(observed, _report_path=str(OBSERVED))
    folds = {(row['participant'], row['held_fold']): row for row in observed['folds']}
    sens.bind_base_support(base, rows, folds, inputs, embedding_index, low_index, check)
    images, source_folds = inputs[person]
    fold = next(item for item in source_folds if item['held_fold'] == held)
    require(fold['primary_reason'] is None, 'Frozen primary support is unavailable')
    indices, directory = fold['indices'], fold['directory']
    frows = np.array([embedding_index[images[int(i)]['image_id']]['output_row'] for i in indices])
    lrows = np.array([low_index[images[int(i)]['image_id']] for i in indices])
    raw = {model: np.asarray(features[model][frows, :], dtype=np.float64) for model in pair}
    model_rdms = {model: native.rdm(raw[model], 'cosine') for model in pair}
    low_rdm = low_module.low_level_rdm(low_features[lrows, :])
    upper = native.upper_triangle(low_rdm)
    require(np.max(upper) != np.min(upper), 'Constant required low-level geometry')
    patterns, neural_rdms = {}, []
    for roi in panel.ROIS[:3]:
        check()
        voxels = np.load(directory / f'{roi}_voxel_indices_c.npy', allow_pickle=False)
        trained = fold['training']['rois'][roi]
        require(voxels.ndim == 1 and voxels.dtype.kind in 'iu' and len(voxels) >= 3
                and len(set(voxels.tolist())) == len(voxels) == trained['retained_voxels']
                == fold['original']['roi_support'][roi]['retained_voxels'], 'Frozen ROI support differs')
        with np.load(Path(trained['per_voxel_values']), allow_pickle=False) as context:
            require(np.array_equal(context['voxel_indices_c'], voxels), 'Frozen voxel identity/order differs')
        patterns[roi] = np.load(directory / f'{roi}_patterns.npy', mmap_mode='r', allow_pickle=False)
        require(patterns[roi].shape == (4712, len(voxels)) and patterns[roi].dtype == np.dtype('float64'),
                'Frozen mean-pattern header differs')
        neural_rdms.append(native.rdm(patterns[roi][indices, :], 'correlation'))
    return images, fold, raw, model_rdms, low_rdm, patterns, neural_rdms


def replay_sensitivities(np, native, report, saved_rows, images, fold, patterns,
                         features, embedding_index, representatives, pair, check):
    person, held = report['participant'], report['held_fold']
    for name in sens.SENSITIVITIES:
        check()
        indices, counts, cells = sens.selection(np, name, images, fold, representatives)
        identity = SENSITIVITIES.parent / f'{person}-fold-{held}-{name}-images.jsonl'
        expected_ids = []
        for i in indices:
            i = int(i)
            image, validity = images[i], fold['validity'][i]
            expected_ids.append({'image_id': image['image_id'], 'pattern_row': i,
                'embedding_row': embedding_index[image['image_id']]['output_row'],
                'source_stratum': image['source_stratum'], 'component_id': image['component_id'],
                'primary_valid_repeats': validity['primary_valid_repeats'],
                'evc_valid_repeats': validity['evc_valid_repeats'], 'intended_repeats': validity['intended_repeats'],
                'observed_metadata_repeats': validity['observed_metadata_repeats']})
        require(panel.saved.json_lines(identity) == expected_ids, 'Original sensitivity image support differs')
        scored = {(roi, model): [] for roi in panel.ROIS[:3] for model in pair}
        for stratum, cell in cells:
            check()
            require(len(cell) >= 3 and (stratum != 'all_sources' or min(counts.values()) >= 3),
                    'Required frozen sensitivity source cell is unavailable')
            frows = np.array([embedding_index[images[int(i)]['image_id']]['output_row'] for i in cell], dtype=np.int64)
            ranks = {model: native.average_ranks(native.upper_triangle(native.rdm(features[model][frows, :], 'cosine')))
                     for model in pair}
            for roi in panel.ROIS[:3]:
                check()
                target = native.average_ranks(native.upper_triangle(native.rdm(patterns[roi][cell, :],
                            'cosine' if name == 'neural_cosine' else 'correlation')))
                for model in pair:
                    r = native.pearson(ranks[model], target)
                    scored[(roi, model)].append({'source_cell': stratum, 'evaluation_images': len(cell),
                        'status': 'evaluable', 'spearman_r': r, 'fisher_z': native.fisher_z(r)})
        for roi in panel.ROIS[:3]:
            for model in pair:
                old = saved_rows[(name, person, held, roi, model)]
                require(old['status'] == 'evaluable' and old['model_id'] == panel.MODELS[model][1]
                        and old['evaluation_images'] == len(indices) and old['source_counts'] == counts
                        and Path(old['evaluation_image_id_file']).resolve() == identity
                        and old['retained_voxels'] == patterns[roi].shape[1], 'Sensitivity row binding differs')
                pieces = scored[(roi, model)]
                require(len(old['source_cell_results']) == len(pieces), 'Sensitivity source-cell count differs')
                for actual, prior in zip(pieces, old['source_cell_results']):
                    require(all(actual[key] == prior[key] for key in ('source_cell', 'evaluation_images', 'status')),
                            'Sensitivity source-cell identity differs')
                    compare(np, [actual['spearman_r'], actual['fisher_z']],
                            [prior['spearman_r'], prior['fisher_z']], 'sensitivity component')
                aggregate = float(np.mean([piece['fisher_z'] for piece in pieces]))
                discrepancy = compare(np, aggregate, old['fisher_z'], 'sensitivity aggregate')
                report['sensitivity_rows'].append({'sensitivity': name, 'roi': roi, 'model_label': model,
                    'fisher_z': aggregate, 'source_cell_results': pieces, 'evaluation_images': len(indices),
                    'source_counts': counts, 'evaluation_image_id_file': str(identity),
                    'maximum_absolute_discrepancy': discrepancy})


def run_unit(np, native, low_module, report, output, decision, reference, check, progress):
    observed, observed_rows, sensitivity_rows, saved = reference
    model = report['selected_model']['model_label']
    vi, mi, di = coordinates(model)
    pair = (model, 'DINOv2')
    person, held = report['participant'], report['held_fold']
    pi = panel.PARTICIPANTS.index(person)
    expected, embedding_index, _, features, low_index, low_features = panel.complete_features(
        np, low_module, nullunit.LOW_LEVEL, check)
    inputs = {p: panel.participant_inputs(np, p, expected, check) for p in panel.PARTICIPANTS}
    images, fold, raw, matrices, low_rdm, patterns, neural = primary_inputs(
        np, native, low_module, expected, embedding_index, features, low_index, low_features,
        inputs, observed, observed_rows, person, held, pair, check)
    report['evaluation_image_id_file'] = str(OBSERVED.parent / f'{person}-fold-{held}-evaluation_images.jsonl')
    report['evaluation_images'] = len(fold['indices'])
    report['source_counts'] = fold['source_counts']
    for ri, roi in enumerate(panel.ROIS[:3]):
        for label in pair:
            check()
            original = observed_rows[(person, held, roi, label)]
            r = native.spearman_rsa(matrices[label], neural[ri])
            partial_r = native.partial_spearman_rsa(matrices[label], neural[ri], low_rdm)
            actual = {'native_spearman_r': r, 'native_fisher_z': native.fisher_z(r),
                      'partial_spearman_r': partial_r, 'partial_fisher_z': native.fisher_z(partial_r)}
            discrepancy = compare(np, list(actual.values()), [original[key] for key in actual], 'primary/partial replay')
            report['primary_rows'].append({'roi': roi, 'model_label': label, **actual,
                                          'maximum_absolute_discrepancy': discrepancy})
    progress('primary_and_partial_complete')
    replay_sensitivities(np, native, report, sensitivity_rows, images, fold, patterns, features,
                         embedding_index, sens.frozen_representatives(expected), pair, check)
    progress('all_four_sensitivities_complete')
    cache = nullunit.rank_cache(np, native, [matrices[label] for label in pair], neural)
    strata = [images[int(i)]['source_stratum'] for i in fold['indices']]
    factors = nullunit.factor_geometry(np, native, raw[model])
    del matrices, low_rdm, patterns, neural, raw, features, low_features, inputs
    permutation = np.lib.format.open_memmap(output / 'permutation_components.npy', mode='w+',
                                            dtype='float64', shape=(PERMUTATIONS, 2, 3))
    geometry = np.lib.format.open_memmap(output / 'geometry_components.npy', mode='w+',
                                        dtype='float64', shape=(GEOMETRIES, 3))
    permutation[:] = np.nan
    geometry[:] = np.nan
    permutation.flush()
    geometry.flush()
    report['component_files'] = {'permutation': 'permutation_components.npy', 'geometry': 'geometry_components.npy'}
    report['component_order'] = list(pair)
    report['random_geometry_model_index'] = vi
    report['maximum_null_discrepancy'] = {'permutation': 0.0, 'geometry': 0.0}
    try:
        for draw in range(PERMUTATIONS):
            check()
            _, z = nullunit.permutation_scores(np, native, cache, strata, draw)
            prior = saved['permutation_component_z'][draw, [mi, di], pi, held, :]
            error = compare(np, z, prior, 'permutation selected/DINO component')
            permutation[draw] = z
            report['completed_permutation_draws'] = draw + 1
            report['maximum_null_discrepancy']['permutation'] = max(error, report['maximum_null_discrepancy']['permutation'])
            if (draw + 1) % 256 == 0:
                permutation.flush()
                progress('permutation_components_retained')
        permutation.flush()
        for draw in range(GEOMETRIES):
            check()
            generated, normalized = nullunit.geometry_from_factors(np, native, factors, vi, draw)
            matrix = 1 - normalized @ normalized.T
            np.fill_diagonal(matrix, 0)
            ranks = native.average_ranks(native.upper_triangle(matrix))
            z = [native.fisher_z(native.pearson(ranks, target)) for target in cache['neural_ranks']]
            error = compare(np, z, saved['geometry_component_z'][draw, vi, pi, held, :], 'random-geometry selected component')
            geometry[draw] = z
            report['completed_geometry_draws'] = draw + 1
            report['maximum_null_discrepancy']['geometry'] = max(error, report['maximum_null_discrepancy']['geometry'])
            del generated, normalized, matrix, ranks
            if (draw + 1) % 32 == 0:
                geometry.flush()
                progress('random_geometry_components_retained')
        check()
    finally:
        permutation.flush()
        geometry.flush()
    require(len(report['primary_rows']) == 6 and len(report['sensitivity_rows']) == 24
            and report['completed_permutation_draws'] == PERMUTATIONS
            and report['completed_geometry_draws'] == GEOMETRIES, 'Whole frozen unit must complete')
    report['complete_unit'] = True
    report['status'] = 'selected_development_rerun_unit_complete'


def selected_gates(np, effects, partial, sensitivities, thresholds):
    gates = {'practical_gain': bool(effects['gain'] >= .02),
             'participant_direction': bool(np.count_nonzero(effects['participant'] > 0) >= 4),
             'participant_influence': bool(np.all(effects['participant_loo'] > 0)),
             'sector_influence': bool(np.all(effects['sector_loo'] > 0)),
             'positive_alignment': bool(effects['unsubtracted_alignment'] > 0),
             'image_null': bool(effects['unsubtracted_alignment'] > thresholds[0]),
             'geometry_null': bool(effects['unsubtracted_alignment'] > thresholds[1]),
             'low_level_partial_rsa': bool(partial['gain'] > 0 and np.all(partial['participant_loo'] > 0))}
    gates.update({name: bool(sensitivities[name]['gain'] > 0) for name in sens.SENSITIVITIES})
    return gates


def replace_null_unit(np, permutations, geometries, model, pi, held, fresh_perm, fresh_geom):
    """Keep both other VLMs while refreshing the selected row and shared DINO."""
    vi, mi, di = coordinates(model)
    require(permutations.shape == (PERMUTATIONS, 4, 5, 5, 3)
            and geometries.shape == (GEOMETRIES, 3, 5, 5, 3)
            and fresh_perm.shape == (PERMUTATIONS, 2, 3) and fresh_geom.shape == (GEOMETRIES, 3),
            'Complete fresh and original null array shapes required')
    require(type(pi) is int and pi in range(5) and type(held) is int and held in range(5), 'Unknown null cell')
    compare(np, fresh_perm[:, 0, :], permutations[:, mi, pi, held, :], 'collected selected permutations')
    compare(np, fresh_perm[:, 1, :], permutations[:, di, pi, held, :], 'collected DINO permutations')
    compare(np, fresh_geom, geometries[:, vi, pi, held, :], 'collected selected geometry')
    permutations[:, mi, pi, held, :] = fresh_perm[:, 0, :]
    permutations[:, di, pi, held, :] = fresh_perm[:, 1, :]
    geometries[:, vi, pi, held, :] = fresh_geom


def claim_output(mode, participant=None, fold=None):
    require(mode in ('unit', 'collect'), 'Unknown rerun operation')
    if mode == 'unit':
        require(participant in panel.PARTICIPANTS and type(fold) is int and fold in range(5), 'Unknown rerun cell')
    output = FAMILY / ('units/' + participant + '-fold-' + str(fold) if mode == 'unit' else 'collection')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # One attempt per exact unit/collection; never overwrite or resume.
    return output


def collect_units(np, native, report, output, decision, reference, check, progress):
    observed, observed_rows, sensitivity_rows, saved = reference
    model = report['selected_model']['model_label']
    vi, mi, di = coordinates(model)
    pair = (model, 'DINOv2')
    values = saved['observed_component_z'].copy()
    partial = saved['observed_partial_component_z'].copy()
    permutations = saved['permutation_component_z'].copy()
    geometries = saved['geometry_component_z'].copy()
    sensitivity = {name: np.array([[[[sensitivity_rows[(name, p, f, r, m)]['fisher_z'] for r in panel.ROIS[:3]]
                     for f in range(5)] for p in panel.PARTICIPANTS] for m in panel.MODELS]) for name in sens.SENSITIVITIES}
    for pi, person in enumerate(panel.PARTICIPANTS):
        for held in range(5):
            check()
            directory = FAMILY / 'units' / f'{person}-fold-{held}'
            path = directory / REPORT_NAME
            unit = fixed_json(path)
            require(unit.get('status') == 'selected_development_rerun_unit_complete'
                    and unit.get('complete_unit') is True and unit.get('errors') == []
                    and unit.get('participant') == person and type(unit.get('held_fold')) is int
                    and unit['held_fold'] == held and unit['selected_model'] == decision['selected_model']
                    and unit['decision_report'] == str(DECISION) and unit['original_panel_deadline_utc'] == report['original_panel_deadline_utc']
                    and unit['first_panel_evaluation_utc'] == decision['first_panel_evaluation_utc']
                    and unit['completed_permutation_draws'] == PERMUTATIONS and unit['completed_geometry_draws'] == GEOMETRIES
                    and unit['component_order'] == list(pair) and unit['random_geometry_model_index'] == vi
                    and unit['component_files'] == {'permutation': 'permutation_components.npy', 'geometry': 'geometry_components.npy'},
                    'Every exact participant/fold unit must complete without substitution')
            start, finish = (datetime.fromisoformat(unit[key]) for key in ('started_utc', 'finished_utc'))
            require(datetime.fromisoformat(decision['finished_utc']) <= start <= finish
                    <= datetime.fromisoformat(report['original_panel_deadline_utc'])
                    and finish <= datetime.now(timezone.utc), 'Rerun unit exceeded the original panel clock')
            original_fold = next(row for row in observed['folds'] if row['participant'] == person and row['held_fold'] == held)
            require(unit['evaluation_image_id_file'] == original_fold['image_id_file']
                    and unit['evaluation_images'] == original_fold['evaluation_images']
                    and unit['source_counts'] == original_fold['source_counts'], 'Unit frozen image support differs')
            primary_rows = {(row['roi'], row['model_label']): row for row in unit['primary_rows']}
            require(len(unit['primary_rows']) == len(primary_rows) == 6
                    and set(primary_rows) == {(r, m) for r in panel.ROIS[:3] for m in pair}, 'Every native/partial row required')
            for ri, roi in enumerate(panel.ROIS[:3]):
                for label, index in zip(pair, (mi, di)):
                    row = primary_rows[(roi, label)]
                    original = observed_rows[(person, held, roi, label)]
                    keys = ('native_spearman_r', 'native_fisher_z', 'partial_spearman_r', 'partial_fisher_z')
                    compare(np, [row[key] for key in keys], [original[key] for key in keys], 'collected native/partial')
                    values[index, pi, held, ri], partial[index, pi, held, ri] = row['native_fisher_z'], row['partial_fisher_z']
            rows = {(r['sensitivity'], r['roi'], r['model_label']): r for r in unit['sensitivity_rows']}
            require(len(unit['sensitivity_rows']) == len(rows) == 24
                    and set(rows) == {(s, r, m) for s in sens.SENSITIVITIES for r in panel.ROIS[:3] for m in pair},
                    'Every required sensitivity row must be present')
            for name in sens.SENSITIVITIES:
                for ri, roi in enumerate(panel.ROIS[:3]):
                    for label, index in zip(pair, (mi, di)):
                        value = rows[(name, roi, label)]['fisher_z']
                        compare(np, value, sensitivity_rows[(name, person, held, roi, label)]['fisher_z'], 'collected sensitivity')
                        sensitivity[name][index, pi, held, ri] = value
            for filename in unit['component_files'].values():
                require(not (directory / filename).is_symlink(), 'Rerun arrays cannot redirect')
            fresh_perm = np.load(directory / 'permutation_components.npy', mmap_mode='r', allow_pickle=False)
            fresh_geom = np.load(directory / 'geometry_components.npy', mmap_mode='r', allow_pickle=False)
            replace_null_unit(np, permutations, geometries, model, pi, held, fresh_perm, fresh_geom)
            report['unit_reports'].append(str(path))
            progress('complete_unit_admitted')
    require(len(report['unit_reports']) == 25, 'All25 units are required before familywise comparison')
    vlm_indices = [list(panel.MODELS).index(label) for label in panel.VLM_LABELS]
    identities = [native.ModelIdentity(panel.MODELS[label][1], panel.MODELS[label][0]) for label in panel.VLM_LABELS]
    thresholds = []
    for family, vlm, dino, threshold_key in (
            ('permutation', permutations[:, vlm_indices], permutations[:, di], 'image_permutation_threshold'),
            ('geometry', geometries, values[di], 'random_geometry_threshold')):
        chosen, statistic, threshold = nullcollector.familywise_distribution(np, native, vlm, dino, identities, check)
        require(np.array_equal(chosen, saved[family + '_selected_VLM_index']), 'Familywise null winning indices changed')
        report[family + '_statistic_maximum_discrepancy'] = compare(
            np, statistic, saved[family + '_signed_statistic'], 'complete three-VLM null statistics')
        report[family + '_threshold_discrepancy'] = compare(
            np, threshold, decision[threshold_key], 'complete three-VLM higher95 threshold')
        report[threshold_key] = threshold
        report[family + '_null_winners_match'] = True
        thresholds.append(threshold)
    def selected_effects(array):
        return {key: value[vi] for key, value in native.panel_effects(array[vlm_indices], array[di]).items()}
    effects, partial_effects = selected_effects(values), selected_effects(partial)
    sensitivity_effects = {name: selected_effects(array) for name, array in sensitivity.items()}
    for key, value in effects.items():
        compare(np, value, decision['effects'][key][vi], 'selected native effect')
    for name, result in {'low_level_partial_rsa': partial_effects, **sensitivity_effects}.items():
        for key, value in result.items():
            compare(np, value, decision['sensitivity_effects'][name][key][vi], 'selected falsifier effect')
    gates = selected_gates(np, effects, partial_effects, sensitivity_effects, thresholds)
    require(gates == decision['gates'][vi] and all(gates.values()), 'Selected-row gate mismatch; no reselection allowed')
    report.update({'complete_rerun': True, 'status': 'selected_development_rerun_verified_pending_configuration_lock',
                   'selected_effects': {key: value.tolist() for key, value in effects.items()},
                   'selected_gates': gates, 'all25_units_complete': True,
                   'all_three_familywise_nulls_reconstructed': True,
                   'all_fresh_components_match_original': True,
                   'existing_required_diagnostics_preserved': decision['diagnostics'],
                   'diagnostics_refitted': False})
    check()


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    worker = commands.add_parser('unit')
    worker.add_argument('--slot', type=int, choices=range(25), required=True)
    worker.add_argument('--max-seconds', type=int, required=True)
    collector = commands.add_parser('collect')
    collector.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    if args.mode == 'unit':
        args.participant, args.fold = slot_cell(args.slot)
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in socket.gethostname()
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(name) == '1' for name in panel.saved.THREAD_ENV),
            'One actual compute CPU and six thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= (14400 if args.mode == 'unit' else 600),
            'Exact EP04 cwd and bounded explicit process limit required')
    decision = fixed_json(DECISION)
    model = admit_decision(decision)
    deadline, wall_end = nullunit.clock_deadline(decision['first_panel_evaluation_utc'], datetime.now(timezone.utc),
                                                time.monotonic(), started + args.max_seconds)
    require(datetime.fromisoformat(wall_end) == datetime.fromisoformat(decision['panel_wall_deadline_utc'])
            and datetime.fromisoformat(decision['finished_utc']) <= datetime.now(timezone.utc), 'Saved decision clock differs')
    output = claim_output(args.mode, getattr(args, 'participant', None), getattr(args, 'fold', None))
    report = {'analysis_id': 'development_rerun', 'mode': args.mode, 'status': 'started',
              '_started': started, 'started_utc': datetime.now(timezone.utc).isoformat(),
              'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(), 'internal_seconds': args.max_seconds,
              'decision_report': str(DECISION), 'selected_model': decision['selected_model'],
              'first_panel_evaluation_utc': decision['first_panel_evaluation_utc'], 'original_panel_deadline_utc': wall_end,
              'source_observed_report': str(OBSERVED), 'source_sensitivity_report': str(SENSITIVITIES),
              'source_null_report': str(NULL_REPORT), 'source_null_scores': str(NULL_SCORES),
              'engineering_atol': ATOL, 'complete_unit': False, 'complete_rerun': False,
              'candidate_decisions_created': 0, 'candidate_reselection_performed': False,
              'candidate_lock_authorized': False, 'OOD_access_authorized': False, 'OOD_outcomes_opened': False,
              'raw_beta_arrays_or_images_opened': False, 'preprocessing_or_diagnostics_refitted': False,
              'nonselected_components': 'Immutable complete original arrays; same3-VLM null competition retained',
              'fresh_DINO_scope': 'Every candidate gain uses fresh DINO permutation/native contributions',
              'automatic_retry': False, 'automatic_successor': False, 'prior_budget_consumption_preserved': True,
              'errors': [], 'primary_rows': [], 'sensitivity_rows': [], 'unit_reports': [],
              'completed_permutation_draws': 0, 'completed_geometry_draws': 0}
    if args.mode == 'unit':
        report.update(slot=args.slot, participant=args.participant, held_fold=args.fold)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Rerun process bound or original120-hour deadline reached')
    def progress(stage):
        report['stage'] = stage
        save_report(output, report)
        print(json.dumps({'stage': stage, 'mode': args.mode, 'participant': report.get('participant'),
                          'fold': report.get('held_fold'), 'permutations': report['completed_permutation_draws'],
                          'geometries': report['completed_geometry_draws']}), flush=True)
        check()
    def interrupted(*_):
        raise TimeoutError('Selected-row rerun interrupted')
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
        progress('decision_bound_before_source_values')
        import numpy as np
        import native_rsa as native
        import low_level_pixels_proposal as low_module
        reference = load_reference(decision, np, check)
        if args.mode == 'unit':
            run_unit(np, native, low_module, report, output, decision, reference, check, progress)
        else:
            collect_units(np, native, report, output, decision, reference, check, progress)
        check()
    except Exception as exc:
        report['status'] = 'technical_selected_development_rerun_failure'
        report['complete_unit'] = report['complete_rerun'] = False
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            report['status'] = 'technical_selected_development_rerun_failure'
            report['complete_unit'] = report['complete_rerun'] = False
            report['errors'].append({'type': 'TimeoutError', 'message': 'Original/process deadline reached at closeout'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_report(output, report)
        if (report['complete_unit'] or report['complete_rerun']) and time.monotonic() >= deadline:
            report['status'] = 'technical_selected_development_rerun_failure'
            report['complete_unit'] = report['complete_rerun'] = False
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_report(output, report)
    return 0 if report['complete_unit'] or report['complete_rerun'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
