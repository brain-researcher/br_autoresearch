#!/usr/bin/env python3
"""Complete one fixed metadata route with visible, bounded source operations.

Uses qualify_group_map_contributors.process_task unchanged. This reads its two
exact source JSON files and enumerates/stats its declared subject effect paths;
it never reads image/header/PNG values. No all-route retry, scoring or dispatch.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import faulthandler
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import signal
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
MANIFEST = EPISODE / 'outputs/runs/source_handoff_qualification/20261003-group-map-metadata02/group_map_metadata_report.json'
DATASET, TASK = 'ds000001', 'balloonanalogrisktask'
INTERNAL_SECONDS = 90
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


class RouteDeadline(RuntimeError):
    """Must not be swallowed by filesystem helpers' normal OSError handling."""


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    hostname = socket.gethostname()
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in hostname
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(name) != '1' for name in THREADS)):
        raise RuntimeError('Actual one-CPU compute step and six thread limits=1 required')
    if Path.cwd().resolve() != EPISODE:
        raise ValueError('Run from the exact EP03 episode directory')
    output = args.output_dir.resolve()
    if output.exists() or not output.is_relative_to(EPISODE / 'outputs/runs/source_handoff_qualification'):
        raise ValueError('Fresh episode-local source qualification output required')
    output.mkdir(parents=True)
    report = {
        'analysis_id': 'source_handoff_qualification',
        'status': 'first_route_join_running', 'route_complete': False,
        'dataset_id': DATASET, 'task': TASK, 'selected_candidate_maps': 13,
        'source_manifest': str(MANIFEST), 'historical_interrupted_job': '46429138',
        'diagnosed_first_route_job': '46658777',
        'enumeration': 'same_two_patterns_name_match_before_directory_check',
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': hostname,
        'allocated_cpus': 1, 'internal_seconds': INTERNAL_SECONDS,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'operations_attempted': 0, 'operations_returned': 0,
        'active_operations': [], 'last_operation': None, 'operation_failures': [],
        'task_result': None, 'errors': [],
        'actual_group_N_authenticated_count': 0,
        'candidate_handoff_qualified': False, 'full_577_program_handoff_equivalence': False,
        'payload_or_png_values_opened': False, 'group_map_payloads_or_headers_opened': False,
        'audit_opened': False, 'scientific_trial': False, 'automatic_retry': False,
        'remaining_routes_launched': False,
    }

    def check():
        if time.monotonic() - started >= INTERNAL_SECONDS:
            raise RouteDeadline('Whole-process 90s first-route deadline reached')

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        pending = output / 'first_route_contributor_support.pending.json'
        pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        pending.replace(output / 'first_route_contributor_support.json')

    def event(phase, operation=None):
        save()
        print(json.dumps({'phase': phase, 'elapsed_seconds': report['elapsed_seconds'],
                          'operation': operation}, allow_nan=False), flush=True)

    def begin(kind, path, pattern=None):
        check()
        report['operations_attempted'] += 1
        operation = {'number': report['operations_attempted'], 'kind': kind, 'path': str(path)}
        if pattern is not None:
            operation['pattern'] = pattern
        report['active_operations'].append(operation)
        event('source_operation_started', operation)
        check()
        return operation

    def finish(operation, error=None):
        if error is None:
            report['operations_returned'] += 1
        else:
            report['operation_failures'].append({**operation, 'error_type': type(error).__name__,
                                                 'error': str(error)})
        report['last_operation'] = operation
        if report['active_operations'] and report['active_operations'][-1] == operation:
            report['active_operations'].pop()
        event('source_operation_returned' if error is None else 'source_operation_failed', operation)
        if error is None:
            check()

    def interrupted(signum, _frame):
        raise RouteDeadline(f'First-route join interrupted by signal {signum}')

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    original = None
    original_path = None
    try:
        for sig in previous_handlers:
            signal.signal(sig, interrupted)
        signal.setitimer(signal.ITIMER_REAL, max(.001, INTERNAL_SECONDS - (time.monotonic() - started)))
        # One traceback can identify a blocked application call even if its I/O
        # does not promptly deliver Python's deadline signal. No repeating timer.
        faulthandler.dump_traceback_later(max(.001, 85 - (time.monotonic() - started)), repeat=False)
        event('before_original_module_import'); check()
        import qualify_group_map_contributors as original
        original_path = original.Path
        event('before_fixed_manifest_read'); check()
        manifest = json.loads(MANIFEST.read_text())
        if (manifest.get('explicit_dataset_task_pairs') != 46
                or manifest.get('candidate_group_Z_filenames') != 297
                or len(manifest.get('tasks', [])) != 46):
            raise ValueError('Original297-map/46-route manifest required')
        selected = manifest['tasks'][0]
        analysis_root = Path(os.environ['OAK']) / 'data/OpenNeuro_analyses/openneuro_fitlins/analyses'
        task_root = analysis_root / DATASET / f'task-{TASK}'
        spec = analysis_root / DATASET / 'code/openneuro_glmfitlins/statsmodel_specs' / DATASET / f'{DATASET}-{TASK}_specs.json'
        if (selected.get('dataset_id') != DATASET or selected.get('task') != TASK
                or selected.get('task_directory') != str(task_root)
                or selected.get('existing_spec_paths') != [str(spec)]
                or len(selected.get('map_filename_candidates', [])) != 13):
            raise ValueError('Only the original first ds000001/BART13-map route is selected')
        allowed_text = {spec, task_root / 'dataset_description.json'}

        class ProgressPath(type(Path())):
            """Forward reads/stats; enumerate the same fixed patterns by name first."""
            def read_text(self, *args, **kwargs):
                if Path(self) not in allowed_text:
                    raise ValueError('Only the two declared source JSON files may be read')
                return self._observed('read_text', super().read_text, *args, **kwargs)

            def is_file(self, *args, **kwargs):
                return self._observed('is_file', super().is_file, *args, **kwargs)

            def is_dir(self, *args, **kwargs):
                return self._observed('is_dir', super().is_dir, *args, **kwargs)

            def _observed(self, kind, call, *args, **kwargs):
                operation = begin(kind, self)
                try:
                    result = call(*args, **kwargs)
                except Exception as exc:
                    finish(operation, exc)
                    raise
                finish(operation)
                return result

            def glob(self, pattern, *args, **kwargs):
                if pattern not in ('sub-*/*_stat-effect_statmap.nii.gz',
                                   'sub-*/ses-*/*_stat-effect_statmap.nii.gz'):
                    raise ValueError('Only the original two subject-node path depths are permitted')
                if args or kwargs:
                    raise ValueError('Only the original POSIX case-sensitive glob call is selected')
                operation = begin('glob', self, pattern)

                def selected_paths(directory, parts):
                    def matching_entries():
                        # Same scandir/name matching as pathlib, but never stat
                        # a nonmatching name just to reject it as a directory.
                        with os.scandir(directory) as entries:
                            return [entry for entry in entries
                                    if fnmatchcase(entry.name, parts[0])]

                    try:
                        entries = directory._observed('scandir_name_filter', matching_entries)
                    except OSError:
                        # Preserve Path.glob's handling of unavailable directories.
                        # RouteDeadline deliberately does not inherit OSError.
                        return
                    for entry in entries:
                        check()
                        path = type(self)(entry.path)
                        if len(parts) == 1:
                            # Like pathlib, yield matching names including broken
                            # links. The unchanged contributor does final is_file.
                            yield path
                            continue
                        try:
                            is_directory = path._observed('matched_directory_is_dir', entry.is_dir)
                        except OSError:
                            continue
                        if is_directory:
                            # Default is_dir follows links, matching the old glob.
                            # Depth is exactly two or three pattern components.
                            yield from selected_paths(path, parts[1:])

                try:
                    if self.is_dir():
                        yield from selected_paths(self, pattern.split('/'))
                except Exception as exc:
                    finish(operation, exc)
                    raise
                finish(operation)

        original.Path = ProgressPath
        event('before_original_process_task'); check()
        result = original.process_task(selected, analysis_root)
        # Retain the useful complete join even if terminal publication later times out.
        report['task_result'] = result
        check()
        maps = result['maps']
        if len(maps) != 13:
            raise ValueError('Selected first route must retain all13 candidate-map rows')
        report['provider_statistic_status_counts'] = dict(Counter(
            row['provider_declared_statistic_status'] for row in maps))
        report['maps_with_source_output_support'] = sum(
            row['source_output_supported_subject_count'] > 0 for row in maps)
        report['route_complete'] = True
        report['status'] = 'first_route_join_completed_actual_group_inclusion_unresolved'
        event('first_route_completed'); check()
    except Exception as exc:
        report['errors'].append({'type': type(exc).__name__, 'message': str(exc)})
        report.update(status='first_route_join_incomplete', route_complete=False)
    finally:
        if original is not None and original_path is not None:
            original.Path = original_path
        faulthandler.cancel_dump_traceback_later()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if time.monotonic() - started >= INTERNAL_SECONDS:
            report.update(status='first_route_join_incomplete', route_complete=False)
            report['errors'].append({'type': 'RouteDeadline', 'message': 'Deadline reached during closure'})
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        if report['route_complete'] and time.monotonic() - started >= INTERNAL_SECONDS:
            report.update(status='first_route_join_incomplete', route_complete=False)
            report['errors'].append({'type': 'RouteDeadline', 'message': 'Deadline reached during final publication'})
            save()
    return 0 if report['route_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
