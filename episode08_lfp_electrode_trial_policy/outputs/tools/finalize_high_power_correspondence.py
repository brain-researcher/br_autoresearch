#!/usr/bin/env python3
"""Finalize the complete saved HF fits after the known legacy-schema failure."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time

from high_power_correspondence import CONTROL, MC_SEEDS, POLICY, VIEWS
from high_power_correspondence_experiment import finalize_scientific_report
from prepare_development_trial_roles import DEVELOPMENT

EXPECTED_FITS = {'outer_training_refits': 760, 'inner_ridge_fits': 15200,
    'capacity_inner_candidate_scores': 76000, 'selector_orderings': 240,
    'direction_permutation_maps': 1600, 'direction_only_models': 40}


def validate_completed_fit(report):
    """Accept only the exact complete, uncharged post-fit legacy-schema failure."""
    if (report.get('status') != 'technical_failure_high_power_correspondence_incomplete'
            or report.get('stage') != 'post_fit_reference_comparison'
            or report.get('error') != "KeyError: 'null_SSE'"
            or report.get('recipe') != 'outputs/high_power_correspondence_proposal.md'
            or report.get('analysis_id') != 'development_search'
            or report.get('policy') != POLICY or report.get('control') != CONTROL
            or report.get('MC_seeds') != list(MC_SEEDS)
            or report.get('held_session_accessed') is not False
            or report.get('held_openings_charged') != 0
            or report.get('valid_evaluations_charged') != 0
            or report.get('scientific_scores_released', False) is not False
            or report.get('named_explanatory_amendment_activated') is not False
            or report.get('raw_arrays_or_decoder_coefficients_written') is not False
            or report.get('decoder_target_labels_shuffled') is not False
            or report.get('input_coordinates_per_electrode') != 2
            or report.get('native_power_guide_pairs') != [[100., 200.], [200., 400.]]
            or report.get('LMP_neural_values_read') is not False
            or report.get('LMP_columns_zero_padded') is not False):
        raise ValueError('Only the qualified complete-fit HF legacy-schema failure may resume')
    if any(report['fit_progress'].get(key) != value for key, value in EXPECTED_FITS.items()):
        raise ValueError('Every adopted fitting/selection count must be complete before reuse')
    if report.get('fitting_trial_read_calls') != 9984 or report.get('evaluation_trials_read') != 320:
        raise ValueError('Complete original fitting/evaluation exposure required')
    if (report.get('completed_session_ids') != list(DEVELOPMENT) or len(report['sessions']) != 8
            or [s['filename'] for s in report['sessions']] != list(DEVELOPMENT)):
        raise ValueError('Exactly all eight completed original sessions required')
    expected = {(n, e) for n in (32, 48, 64, 80, 96) for e in (8, 16, 32, 96)}
    for session in report['sessions']:
        if (session['animal'] != ('Mihili' if session['filename'].startswith('Mihili') else 'Chewie')
                or session.get('all95_target_models_fixed_before_evaluation') is not True
                or session.get('all30_target_orders_fixed_before_post_initial_reads') is not True
                or session['target_unit_count'] < 1 or set(session['panels']) != {POLICY, CONTROL}
                or [s['mc_seed'] for s in session['seed_panels']] != list(MC_SEEDS)):
            raise ValueError('Original animal, complete branch and information-barrier flags required')
        roles = session['source_roles']
        channels, initial, fitting, evaluation = (roles[k] for k in
            ('physical_channels', 'initial_trial_ids', 'fitting_trial_ids', 'evaluation_trial_ids'))
        if (len(channels) != 96 or channels != sorted(set(channels))
                or len(initial) != 32 or len(set(initial)) != 32
                or len(fitting) != 96 or len(set(fitting)) != 96 or not set(initial) <= set(fitting)
                or len(evaluation) != 40 or len(set(evaluation)) != 40 or set(fitting) & set(evaluation)
                or len(roles['initial_inner_folds']) != 4):
            raise ValueError('Complete original disjoint trial/electrode roles required')
        for fold in roles['initial_inner_folds']:
            fit, valid = fold['fitting_trials'], fold['validation_trials']
            if (len(fit) != 24 or len(set(fit)) != 24 or len(valid) != 8 or len(set(valid)) != 8
                    or set(fit) & set(valid) or set(fit) | set(valid) != set(initial)):
                raise ValueError('Exact original initializer inner views required')
        selections = session['selection_records']
        if (len(selections) != 30 or {(r['mc_seed'], r['view']) for r in selections}
                != {(seed, view) for seed in (None, *MC_SEEDS) for view in VIEWS}):
            raise ValueError('All thirty recorded initializer orders required')
        for panel in [*session['panels'].values(), *(s['panel'] for s in session['seed_panels'])]:
            if (len(panel['cells']) != 20 or
                    {(c['training_trials'], c['electrodes']) for c in panel['cells']} != expected):
                raise ValueError('Exact twenty saved score cells required')
        for panel in [session['panels'][POLICY], *(s['panel'] for s in session['seed_panels'])]:
            if len(panel['fitted_models']) != 20:
                raise ValueError('Every saved candidate/seed fitted-score panel must be complete')
    # Exact equality to qualified historical role records is enforced again
    # by attach_references before any result is released; no source is opened.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('completed-fit-report', 'reference-report', 'lmp-reference', 'output-dir'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Explicit single-thread numerical settings required')
    if args.output_dir.exists() or not 0 < args.max_seconds <= 60:
        raise ValueError('Fresh preserved output and finalization bound at most60 seconds required')
    started, cpu_started = time.monotonic(), time.process_time()
    def deadline(*_):
        raise TimeoutError('Selected finalization-only deadline reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    try:
        original = json.loads(args.completed_fit_report.read_text())
        validate_completed_fit(original)
    except BaseException:
        signal.alarm(0)
        raise
    report = copy.deepcopy(original)
    report['reused_fit_report'] = str(args.completed_fit_report)
    report['reused_fit_report_retained_bytes'] = original['retained_report_bytes']
    report['original_post_fit_failure'] = original['error']
    report.pop('error', None)
    report['reference_paths'] = {'reference_report': str(args.reference_report),
                                 'lmp_reference': str(args.lmp_reference)}
    report['finalization_only'] = True
    report['new_target_fits'] = 0
    report['new_scientific_source_reads'] = 0
    report['original_fit_costs_preserved'] = True
    report['finalization_started_utc'] = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        previous = json.loads(args.reference_report.read_text())
        lmp = json.loads(args.lmp_reference.read_text())
        finalize_scientific_report(report, previous, lmp)
    except Exception as error:
        report.update(status='technical_failure_high_power_finalization_incomplete',
            scientific_scores_released=False, valid_evaluations_charged=0,
            error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report['finalization_resources'] = {'slurm_job_id': os.environ['SLURM_JOB_ID'],
            'hostname': socket.gethostname(), 'elapsed_seconds': time.monotonic()-started,
            'process_CPU_seconds': time.process_time()-cpu_started,
            'maximum_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'new_target_fits': 0, 'new_scientific_source_reads': 0,
            'finalized_utc': datetime.now(timezone.utc).isoformat()}
        report['retained_report_bytes'] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
            size = len(payload.encode('utf-8'))
            if report['retained_report_bytes'] == size:
                break
            report['retained_report_bytes'] = size
        (args.output_dir/'high_power_correspondence.json').write_text(payload)
    print(json.dumps({key: report[key] for key in
        ('status', 'finalization_only', 'new_target_fits', 'valid_evaluations_charged', 'finalization_resources')}), flush=True)


if __name__ == '__main__':
    main()
