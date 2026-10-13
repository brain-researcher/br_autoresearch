#!/usr/bin/env python3
"""Collect explicitly named complete EP04 null units, without candidate selection.

All 25 participant/fold cells and every registered draw must be present exactly
once before any familywise statistic or threshold is computed. Inputs are the
fixed observed report, its saved image identities and named unit JSON/JSONL files;
no raw source, feature array, directory discovery or incomplete-prefix estimate.
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

import run_registered_null_unit as unit


panel = unit.panel
EPISODE = panel.EPISODE
PERMUTATIONS = 10000
GEOMETRIES = 1000
MAX_SECONDS = 600
REPORT_NAME = 'registered_null_collection.json'


def require(condition, message):
    panel.require(condition, message)


def declared_range(value, limit):
    require(isinstance(value, list) and len(value) == 2
            and all(type(item) is int for item in value)
            and 0 <= value[0] <= value[1] <= limit, 'Invalid half-open registered draw range')
    return tuple(value)


def claim_range(coverage, person, fold, interval):
    start, stop = interval
    require(not coverage[person, fold, start:stop].any(),
            'Overlapping declared ranges would count a participant/fold/draw twice')
    coverage[person, fold, start:stop] = True


def require_complete_coverage(permutation_coverage, geometry_coverage):
    require(permutation_coverage.all() and geometry_coverage.all(),
            'Every registered draw requires all 25 participant/fold contributions; no partial threshold')


def scores(np, native, row, shape):
    correlations = native.finite_array(row['spearman_r'])
    z = native.finite_array(row['fisher_z'])
    require(correlations.shape == z.shape == shape and np.all(np.abs(correlations) < 1),
            'Wrong score shape or undefined Fisher correlation')
    expected = np.array([native.fisher_z(float(value)) for value in correlations.flat]).reshape(shape)
    require(np.max(np.abs(expected-z)) <= unit.ENGINEERING_ATOL,
            'Saved Fisher scores differ from their signed correlations')
    return z


def observed_values(np, native, record, keyed):
    values = np.empty((4, 5, 5, 3), dtype=np.float64)
    partial = np.empty_like(values)
    for model_index, model in enumerate(unit.MODEL_ORDER):
        for person_index, person in enumerate(panel.PARTICIPANTS):
            for fold in range(5):
                for roi_index, roi in enumerate(unit.ROI_ORDER):
                    row = keyed[(person, fold, roi, model)]
                    require(row['model_id'] == panel.MODELS[model][1]
                            and row['native_status'] == row['partial_status'] == 'evaluable',
                            'Observed primary identity or evaluability differs')
                    for prefix, destination in (('native', values), ('partial', partial)):
                        checked = scores(np, native, {
                            'spearman_r': [row[f'{prefix}_spearman_r']],
                            'fisher_z': [row[f'{prefix}_fisher_z']]}, (1,))
                        destination[model_index, person_index, fold, roi_index] = checked[0]
    require([item['model_label'] for item in record['models']] == list(unit.MODEL_ORDER),
            'Observed released model ordering differs')
    return values, partial


def validate_unit(np, record, path, observed, cells, values, wall_end):
    person, fold = record['participant'], record['held_fold']
    require(person in panel.PARTICIPANTS and type(fold) is int and fold in range(5),
            'Unit participant/fold is outside the fixed panel')
    person_index = panel.PARTICIPANTS.index(person)
    require(record.get('analysis_id') == 'controls_diagnostics'
            and record.get('mode') == 'registered_null_participant_fold_unit'
            and record.get('status') == 'completed_selected_null_ranges'
            and all(record.get(key) is True for key in (
                'complete_selected_ranges', 'source_context_validated',
                'signed_finite_null_values_preserved', 'fixed_DINO_for_geometry_null',
                'external_range_selection_only'))
            and all(record.get(key) is False for key in (
                'complete_registered_nulls', 'local_winner_or_threshold_computed',
                'candidate_selection_performed', 'candidate_lock_authorized',
                'new_preprocessing_or_neural_fit', 'raw_beta_arrays_or_images_opened',
                'excluded_outcomes_opened', 'automatic_retry', 'automatic_resume'))
            and record.get('errors') == [] and record.get('current_attempt') is None,
            f'Only completed unselected null units are eligible: {path}')
    require(Path(record['observed_report']).resolve() == unit.OBSERVED
            and Path(record['low_level_report']).resolve() == unit.LOW_LEVEL/'low_level_preparation_report.json'
            and record['first_panel_evaluation_utc'] == observed['first_panel_evaluation_utc']
            and record['panel_wall_deadline_utc'] == wall_end,
            'Observed input or original panel clock differs')
    start, end = (datetime.fromisoformat(record[key]) for key in ('started_utc', 'finished_utc'))
    require(start.tzinfo is not None and end.tzinfo is not None
            and datetime.fromisoformat(observed['finished_utc']) <= start <= end
            <= datetime.fromisoformat(wall_end) and end <= datetime.now(timezone.utc),
            'Unit timestamps precede completed observation or exceed the original panel clock')
    require(record['model_order'] == list(unit.MODEL_ORDER)
            and record['random_VLM_order'] == list(unit.VLM_ORDER)
            and record['ROI_order'] == list(unit.ROI_ORDER)
            and record['models'] == observed['models']
            and record['registered_permutation_draws'] == PERMUTATIONS
            and record['registered_geometry_draws_per_VLM'] == GEOMETRIES
            and record['permutation_seed'] == 'PCG64(draw_index)'
            and record['random_seed'] == 'PCG64(100000+1000*model_index+draw_index)'
            and record['random_X_scope'] == 'Exact eligible evaluation-fold raw release embedding matrix before downstream L2'
            and record['permutation_unit'] == 'Whole image within registered source stratum; same bijection for all models and sectors',
            'Frozen null model, seed, image-level or geometry convention differs')
    cell = cells[(person, fold)]
    require(Path(record['evaluation_image_file']).resolve() == cell['path']
            and record['image_order'] == cell['image_ids']
            and record['evaluation_images'] == cell['count']
            and record['source_counts'] == cell['source_counts']
            and Path(record['preprocessing_report']).resolve()
            == Path(observed['preprocessing_reports'][person]).resolve()
            and Path(record['training_context_report']).resolve()
            == Path(observed['training_context_reports'][person]).resolve(),
            'Exact observed image support or frozen preprocessing binding differs')
    require(record['observed_replay_atol'] == unit.ENGINEERING_ATOL
            and type(record['observed_replay_maximum_absolute_discrepancy']) in (int, float)
            and 0 <= record['observed_replay_maximum_absolute_discrepancy'] <= unit.ENGINEERING_ATOL,
            'Unit source replay was not within the unchanged arithmetic tolerance')
    fixed = record['fixed_observed_fisher_z']
    require(set(fixed) == set(unit.MODEL_ORDER), 'Unit fixed observed model support differs')
    for model_index, model in enumerate(unit.MODEL_ORDER):
        actual = np.asarray(fixed[model], dtype=np.float64)
        require(actual.shape == (3,) and np.isfinite(actual).all()
                and np.max(np.abs(actual-values[model_index, person_index, fold])) <= unit.ENGINEERING_ATOL,
                'Unit fixed observed scores differ from the exact completed panel')
    permutation = declared_range(record['permutation_range'], PERMUTATIONS)
    geometry = declared_range(record['geometry_range'], GEOMETRIES)
    require(permutation[0] < permutation[1] or geometry[0] < geometry[1], 'Empty unit contributes no draws')
    require(record['completed_permutation_draws'] == permutation[1]-permutation[0]
            and record['completed_geometry_by_model']
            == {model: geometry[1]-geometry[0] for model in unit.VLM_ORDER}
            and record['score_file'] == 'draw_scores.jsonl', 'Declared unit counts/file differ')
    score_path = (path.parent/'draw_scores.jsonl').resolve()
    require(score_path.parent == path.parent and score_path.is_file(), 'Unit score file is missing or redirected')
    return person_index, fold, permutation, geometry, score_path


def collect_scores(np, native, plans, permutation_values, geometry_values, check):
    for person, fold, permutation, geometry, path in plans:
        seen_permutation, seen_geometry = set(), set()
        with path.open() as stream:
            for ordinal, line in enumerate(stream):
                if ordinal % 64 == 0:
                    check()
                record = json.loads(line)
                draw = record['draw_index']
                require(type(draw) is int and record['ROI_order'] == list(unit.ROI_ORDER),
                        'Score draw or ROI ordering differs')
                if record['family'] == 'whole_image_permutation':
                    require(permutation[0] <= draw < permutation[1] and draw not in seen_permutation
                            and record['model_order'] == list(unit.MODEL_ORDER),
                            'Permutation score is duplicated, outside its declared range or misordered')
                    permutation_values[draw, :, person, fold, :] = scores(np, native, record, (4, 3))
                    seen_permutation.add(draw)
                elif record['family'] == 'covariance_matched_random_geometry':
                    model = record['model_index']
                    require(type(model) is int and model in range(3)
                            and geometry[0] <= draw < geometry[1] and (draw, model) not in seen_geometry
                            and record['model_label'] == unit.VLM_ORDER[model]
                            and record['seed'] == 100000+1000*model+draw,
                            'Random-geometry component is duplicated, misidentified or outside its range')
                    geometry_values[draw, model, person, fold, :] = scores(np, native, record, (3,))
                    seen_geometry.add((draw, model))
                else:
                    raise ValueError('Unknown null family in a selected unit')
        require(seen_permutation == set(range(*permutation))
                and seen_geometry == {(draw, model) for draw in range(*geometry) for model in range(3)},
                f'Unit score contents do not cover every declared component: {path}')
        check()


def familywise_distribution(np, native, vlm_values, dino_values, identities, check):
    """No masks or eligibility gates enter the unconditional three-row selection."""
    count = len(vlm_values)
    require(vlm_values.shape == (count, 3, 5, 5, 3)
            and dino_values.shape in ((5, 5, 3), (count, 5, 5, 3))
            and np.isfinite(vlm_values).all() and np.isfinite(dino_values).all(),
            'Familywise statistics require complete finite five-person/five-fold/three-sector panels')
    selected = np.empty(count, dtype=np.int8)
    statistics = np.empty(count, dtype=np.float64)
    for draw in range(count):
        if draw % 64 == 0:
            check()
        dino = dino_values if dino_values.ndim == 3 else dino_values[draw]
        selected[draw], statistics[draw] = native.familywise_null_statistic(
            vlm_values[draw], dino, identities)
    check()
    return selected, statistics, float(np.quantile(statistics, .95, method='higher'))


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--unit-reports', nargs='+', type=Path)
    inputs.add_argument('--report-list', type=Path, help='JSON array of explicit report paths, relative to EP04 cwd if needed')
    parser.add_argument('--observed-report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args(argv)
    host = socket.gethostname()
    require(os.environ.get('SLURM_JOB_ID') and '-ln' not in host
            and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
            and all(os.environ.get(name) == '1' for name in panel.saved.THREAD_ENV),
            'One actual compute CPU and six numerical thread limits=1 required')
    require(Path.cwd().resolve() == EPISODE and 0 < args.max_seconds <= MAX_SECONDS,
            'Use EP04 cwd and an explicit positive process cap<=600s')
    output, observed_path = args.output_dir.resolve(), args.observed_report.resolve()
    require(not output.exists() and output.is_relative_to(EPISODE/'outputs/runs/registered_null_collection')
            and observed_path == unit.OBSERVED, 'Fresh collector output and exact observed01 report required')
    output.mkdir(parents=True)
    deadline = started+args.max_seconds
    report = {'analysis_id': 'controls_diagnostics', 'mode': 'complete_registered_null_collection',
        'status': 'started', 'complete_registered_nulls': False,
        'complete_declared_coverage': False, 'complete_score_coverage': False,
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': host,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'internal_seconds': args.max_seconds,
        'observed_report': str(observed_path), 'unit_reports': [], 'errors': [],
        'candidate_screen_performed': False, 'candidate_selection_performed': False,
        'candidate_lock_authorized': False, 'encoding_or_sensitivities_declared_complete': False,
        'raw_source_or_feature_arrays_opened': False, 'directory_discovery': False,
        'automatic_retry': False, 'automatic_resume': False,
        'registered_permutation_draws': PERMUTATIONS, 'registered_geometry_draws_per_VLM': GEOMETRIES,
        'VLM_order': list(unit.VLM_ORDER), 'model_order': list(unit.MODEL_ORDER),
        'participant_order': list(panel.PARTICIPANTS), 'fold_order': list(range(5)),
        'ROI_order': list(unit.ROI_ORDER), 'signed_null_statistics_preserved': True,
        'null_selection': 'unconditional argmax G_dev over all three VLMs after all 25 cells; no gates',
        'null_statistic': 'selected VLM unsubtracted equal-participant/fold/sector Fisher-z alignment',
        'quantile': .95, 'quantile_method': 'higher', 'comparison_rule': 'strictly_greater_than',
        'positive_observed_alignment_required_separately': True}

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Collector process bound or original120-hour panel deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_KiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        temporary = output/'registered_null_collection.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/REPORT_NAME)

    def interrupted(*_):
        raise TimeoutError('Null collection interrupted or deadline reached')

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        save()
        observed = panel.read_json(observed_path)
        keyed = unit.validate_observed(observed, observed_path)
        deadline, wall_end = unit.clock_deadline(observed['first_panel_evaluation_utc'], datetime.now(timezone.utc),
                                               time.monotonic(), deadline)
        report['first_panel_evaluation_utc'] = observed['first_panel_evaluation_utc']
        report['panel_wall_deadline_utc'] = wall_end
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
        selected_paths = args.unit_reports if args.report_list is None else panel.read_json(args.report_list)
        require(isinstance(selected_paths, list) and selected_paths
                and all(isinstance(path, (str, Path)) for path in selected_paths),
                'Provide a nonempty explicit list of unit report paths')
        paths = [Path(path).resolve() for path in selected_paths]
        require(len(paths) == len(set(paths)) and all(path.name == unit.REPORT_NAME
                and path.is_relative_to(EPISODE/'outputs/runs/registered_nulls') and path.is_file()
                for path in paths), 'Duplicate, absent or outside-scope unit report path')
        report['unit_reports'] = [str(path) for path in paths]
        import numpy as np
        import native_rsa as native
        values, partial = observed_values(np, native, observed, keyed)
        cells = {}
        for fold in observed['folds']:
            check()
            person, held = fold['participant'], fold['held_fold']
            path = (observed_path.parent/f'{person}-fold-{held}-evaluation_images.jsonl').resolve()
            require(Path(fold['image_id_file']).resolve() == path and path.parent == observed_path.parent,
                    'Observed image identity path differs')
            records = panel.saved.json_lines(path)
            identifiers = [row['image_id'] for row in records]
            require(len(identifiers) == len(set(identifiers)) == fold['evaluation_images']
                    and {source: sum(row['source_stratum'] == source for row in records)
                         for source in panel.STRATA} == fold['source_counts'],
                    'Observed exact image identity count or source composition differs')
            cells[(person, held)] = {'path': path, 'image_ids': identifiers,
                'count': len(identifiers), 'source_counts': fold['source_counts']}
        permutation_coverage = np.zeros((5, 5, PERMUTATIONS), dtype=bool)
        geometry_coverage = np.zeros((5, 5, GEOMETRIES), dtype=bool)
        plans = []
        for path in paths:
            check()
            plan = validate_unit(np, panel.read_json(path), path, observed, cells, values, wall_end)
            person, fold, permutation, geometry, _ = plan
            claim_range(permutation_coverage, person, fold, permutation)
            claim_range(geometry_coverage, person, fold, geometry)
            plans.append(plan)
        require_complete_coverage(permutation_coverage, geometry_coverage)
        report['complete_declared_coverage'] = True
        save()
        permutation_values = np.full((PERMUTATIONS, 4, 5, 5, 3), np.nan)
        geometry_values = np.full((GEOMETRIES, 3, 5, 5, 3), np.nan)
        collect_scores(np, native, plans, permutation_values, geometry_values, check)
        require(np.isfinite(permutation_values).all() and np.isfinite(geometry_values).all(),
                'A required participant/fold/model/ROI/draw score is absent')
        report['complete_score_coverage'] = True
        vlm_indices = [unit.MODEL_ORDER.index(model) for model in unit.VLM_ORDER]
        dino_index = unit.MODEL_ORDER.index('DINOv2')
        identities = [native.ModelIdentity(panel.MODELS[model][1], panel.MODELS[model][0])
                      for model in unit.VLM_ORDER]
        perm_selected, perm_stat, perm_threshold = familywise_distribution(
            np, native, permutation_values[:, vlm_indices], permutation_values[:, dino_index], identities, check)
        geom_selected, geom_stat, geom_threshold = familywise_distribution(
            np, native, geometry_values, values[dino_index], identities, check)
        report['image_permutation_threshold'] = perm_threshold
        report['random_geometry_threshold'] = geom_threshold
        effects = native.panel_effects(values[vlm_indices], values[dino_index])
        partial_effects = native.panel_effects(partial[vlm_indices], partial[dino_index])
        report['all_three_observed_screen_inputs'] = [
            {'model_label': model, 'model_id': panel.MODELS[model][1], 'dimension': panel.MODELS[model][0],
             'native': {key: value[index].tolist() for key, value in effects.items()},
             'low_level_partial': {key: value[index].tolist() for key, value in partial_effects.items()}}
            for index, model in enumerate(unit.VLM_ORDER)]
        check()
        np.savez(output/'complete_null_scores.npz',
                 permutation_component_z=permutation_values, geometry_component_z=geometry_values,
                 observed_component_z=values, observed_partial_component_z=partial,
                 permutation_selected_VLM_index=perm_selected, permutation_signed_statistic=perm_stat,
                 geometry_selected_VLM_index=geom_selected, geometry_signed_statistic=geom_stat)
        report['complete_score_arrays'] = 'complete_null_scores.npz'
        check()
        report['complete_registered_nulls'] = True
        report['status'] = 'complete_registered_nulls_without_candidate_decision'
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report['complete_registered_nulls'] = False
        report['status'] = 'incomplete_registered_null_collection'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            report['complete_registered_nulls'] = False
            report['status'] = 'incomplete_registered_null_collection'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during finalization'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['complete_registered_nulls'] and time.monotonic() >= deadline:
            report['complete_registered_nulls'] = False
            report['status'] = 'incomplete_registered_null_collection'
            report['errors'].append({'type': 'TimeoutError', 'message': 'Deadline reached during final publication'})
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save()
    return 0 if report['complete_registered_nulls'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
