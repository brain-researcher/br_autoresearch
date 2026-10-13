#!/usr/bin/env python3
"""Finite X-only logdet selection with true-target heads and reused references."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import socket
import time
import numpy as np

from acquired_only_reader import Exposure, read_trials
from development_learned_electrodes import role_record
from development_reader import selected_metadata
from initializer_logdet import POLICY
from initializer_logdet_experiment import (UNIFORM, REPEATABILITY, MC_SEEDS, aggregate,
                                          attach_references, fit_target, score_target)
from prepare_development_trial_roles import DEVELOPMENT
from ridge_training import new_progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('source-root', 'metadata-report', 'trial-roles', 'reference-report', 'output-dir'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Explicit single-thread numerical settings required')
    if args.output_dir.exists() or not 0 < args.max_seconds <= 180:
        raise ValueError('Fresh preserved attempt and deadline at most180 seconds required')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    def deadline(*_):
        raise TimeoutError('Selected initializer logdet full-cohort deadline reached')
    def check():
        if time.monotonic()-started >= args.max_seconds:
            deadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.max_seconds)
    progress = new_progress()
    progress.update(selector_orderings=0, logdet_selection_steps=0, selection_seconds=0.,
        numerical_negative_gain_occurrences=0, capacity_inner_candidate_scores=0, direction_only_models=0)
    report = {'analysis_id': 'development_search', 'recipe': 'outputs/initializer_logdet_proposal.md',
        'status': 'running', 'policy': POLICY, 'reference_uniform_policy': UNIFORM,
        'reference_repeatability_policy': REPEATABILITY, 'reference_uniform_seeds': list(MC_SEEDS),
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'fit_progress': progress,
        'sessions': [], 'completed_session_ids': [], 'stage': 'new_target_fits',
        'logical_neural_bytes': 0, 'unique_logical_neural_bytes': 0,
        'fitting_trial_read_calls': 0, 'evaluation_trials_read': 0,
        'valid_evaluations_charged': 0, 'held_openings_charged': 0, 'held_session_accessed': False,
        'global_search_completed': False, 'named_explanatory_amendment_activated': False,
        'raw_arrays_or_decoder_coefficients_written': False,
        'selector_target_values_used': False, 'decoder_target_labels_shuffled': False,
        'target_definition': 'one ten-bin time mean per original native unit; no target averaging',
        'selector_objective': 'X-only alpha1 mean-loss logdet group-volume gain; exactly32 greedy groups',
        'input_coordinates_per_electrode': 3,
        'equal_initializer_and_feature_exposure_vs_references': True,
        'targets_acquired_and_used_for_decoding': True,
        'boundary': 'trusted acquired-only views; same UID does not provide OS isolation',
        'claim_limit': 'fixed target-free diversity policy; no removal-value, biological complementarity or joint acquisition claim'}
    exposure = Exposure(report)
    try:
        metadata, roles = json.loads(args.metadata_report.read_text()), json.loads(args.trial_roles.read_text())
        if len(roles['sessions']) != 8 or {s['filename'] for s in roles['sessions']} != set(DEVELOPMENT):
            raise ValueError('All eight original development roles required')
        sources = {name: selected_metadata(metadata, roles, name) for name in DEVELOPMENT}
        for filename in DEVELOPMENT:
            check()
            source, role, records = sources[filename]
            before_models, before_orders = progress['outer_training_refits'], progress['selector_orderings']
            t = time.monotonic()
            models, nulls, guide, selections = fit_target(args.source_root/filename, source, role, records,
                                                         progress, check, exposure)
            if (progress['outer_training_refits']-before_models != 20
                    or progress['selector_orderings']-before_orders != 5):
                raise ArithmeticError('All20 target models and5 selector orders required before evaluation')
            fitted_at = time.monotonic()
            ex, ey, _ = read_trials(args.source_root/filename, source, role, records,
                role['evaluation_trials'], source['common_physical_channels'], check, exposure, guide, evaluation=True)
            labels = np.asarray([records[t]['direction_index'] for t in role['evaluation_trials']])
            panel = score_target(models, nulls, ex, ey, labels, source['common_physical_channels'])
            report['sessions'].append({'filename': filename, 'animal': role['animal'],
                'panels': {POLICY: panel}, 'selection_records': selections,
                'source_roles': role_record(source, role), 'target_unit_count': ey.shape[1],
                'all20_target_models_fixed_before_evaluation': True,
                'all5_orders_fixed_before_post_initial_reads': True,
                'training_read_and_fit_seconds': fitted_at-t, 'elapsed_seconds': time.monotonic()-t})
            report['completed_session_ids'].append(filename)
            print(json.dumps({'session_completed': filename, 'elapsed_seconds': time.monotonic()-t,
                              'actual_target_models': 20, 'scientific_scores_released': False}), flush=True)
            del models, nulls, guide, selections, ex, ey
        expected = {'outer_training_refits': 160, 'inner_ridge_fits': 3200,
                    'capacity_inner_candidate_scores': 16000, 'selector_orderings': 40,
                    'logdet_selection_steps': 1280, 'direction_only_models': 40}
        if (any(progress[k] != v for k, v in expected.items())
                or report['fitting_trial_read_calls'] != 2304 or report['evaluation_trials_read'] != 320):
            raise ArithmeticError('Complete adopted target/selector/read counts differ')
        check()
        report['stage'] = 'post_fit_reference_comparison'
        reference = json.loads(args.reference_report.read_text())
        replay, direction_replay = attach_references(report['sessions'], reference)
        report['historical_full96_score_replay_max_absolute_difference'] = replay
        report['historical_direction_null_Q_replay_max_absolute_difference'] = direction_replay
        report['panels'] = aggregate(report['sessions'])
        report['fixed_prediction_session_omission'] = [
            {'excluded_session': name, 'panels': aggregate([s for s in report['sessions'] if s['filename'] != name], False)}
            for name in DEVELOPMENT]
        candidate = next(p for p in report['panels'] if p['policy'] == POLICY)
        report['prospective_directional_screens'] = {
            'candidate_positive_vs_uniform_in_both_animals':
                all(a['mean_delta_Q_from_'+UNIFORM] > 0 for a in candidate['animals']),
            'candidate_positive_vs_marginal_in_both_animals':
                all(a['mean_delta_Q_from_initial_marginal'] > 0 for a in candidate['animals']),
            'significance_or_promotion_claim': False}
        report['uniform_seed_summaries'] = []
        for seed in MC_SEEDS:
            animals = []
            for animal in ('Chewie', 'Mihili'):
                sessions = [s for s in report['sessions'] if s['animal'] == animal]
                seed_q = float(np.mean([next(p['panel']['mean_Q'] for p in s['uniform_seed_panels']
                                             if p['mc_seed'] == seed) for s in sessions]))
                candidate_q = float(np.mean([s['panels'][POLICY]['mean_Q'] for s in sessions]))
                animals.append({'animal': animal, 'uniform_Q': seed_q, 'candidate_minus_seed_Q': candidate_q-seed_q})
            report['uniform_seed_summaries'].append({'mc_seed': seed, 'animals': animals,
                'candidate_minus_seed_Q': float(np.mean([a['candidate_minus_seed_Q'] for a in animals]))})
        report['reference_reuse'] = {'reference_report': str(args.reference_report),
            'read_only_after_all_new_fits': True, 'new_reference_fits': 0, 'new_reference_charges': 0,
            'old_serialized_source_role_identity_verified': True,
            'old_native_unit_guide_identity_claimed': False,
            'legacy_marginal_SSE_denominator_replay_claimed': False,
            'recorded_target_denominator_reference_policies': [REPEATABILITY, UNIFORM]}
        report.update(status='completed_one_initializer_logdet_comparison', stage='complete',
            valid_evaluations_charged=1, valid_evaluations_total=35,
            search_progress_resolution_Q=.001, scientific_scores_released=True)
    except Exception as error:
        report.update(status='technical_failure_initializer_logdet_incomplete', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        signal.alarm(0)
        report.update(elapsed_seconds=time.monotonic()-started, process_CPU_seconds=time.process_time()-cpu_started,
            maximum_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            observed_utc=datetime.now(timezone.utc).isoformat())
        report['retained_report_bytes'] = 0
        for _ in range(4):
            payload = json.dumps(report, indent=2, allow_nan=False)+'\n'
            size = len(payload.encode('utf-8'))
            if report['retained_report_bytes'] == size:
                break
            report['retained_report_bytes'] = size
        (args.output_dir/'initializer_logdet.json').write_text(payload)
    print(json.dumps({key: report[key] for key in
        ('status', 'elapsed_seconds', 'maximum_RSS_KiB', 'valid_evaluations_charged')}), flush=True)


if __name__ == '__main__':
    main()
