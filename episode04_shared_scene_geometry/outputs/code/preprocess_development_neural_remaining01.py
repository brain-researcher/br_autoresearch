#!/usr/bin/env python3
"""Finite two-worker continuation of the unchanged EP04 preprocessing pilot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


PARTICIPANTS = ('sub-01', 'sub-03', 'sub-06', 'sub-07')
EPISODE = 'episode04_shared_scene_geometry'
INTERNAL_SECONDS = 2040
CHILD_EXTERNAL_SECONDS = 900
CHILD_INTERNAL_SECONDS = 840
PER_PARTICIPANT_BYTES = 16 * 1024 ** 3


def read_json(path):
    return json.loads(path.read_text())


def main():
    if (not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname()
            or int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < 2):
        raise RuntimeError('Verified Slurm compute allocation with at least two CPUs required')
    episode = Path(os.environ['OAK']) / 'users/zijiao/br_autoresearch' / EPISODE
    if Path.cwd().resolve() != episode.resolve():
        raise ValueError('Run from the named EP04 episode')
    payload_root = Path(os.environ['SCRATCH']) / 'br_autoresearch' / EPISODE / 'neural_preprocessing01'
    output_root = Path('outputs/runs/neural_preprocessing/20261003-remaining-folds01')
    source_root = Path('outputs/runs/source_readiness/20261003-development_neural_full01')
    report_path = output_root / 'bundle.json'
    if report_path.exists():
        raise ValueError('Preserve the existing bundle; no automatic retry or overwrite')
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        'analysis_id': 'neural_preprocessing', 'participants': list(PARTICIPANTS),
        'job_id': os.environ['SLURM_JOB_ID'], 'hostname': socket.gethostname(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'complete_remaining_participants': False, 'no_retry': True,
        'max_concurrent_children': 2, 'numeric_threads_per_child': 1,
        'internal_envelope_seconds': INTERNAL_SECONDS,
        'child_external_seconds': CHILD_EXTERNAL_SECONDS,
        'child_internal_seconds': CHILD_INTERNAL_SECONDS,
        'per_participant_output_cap_bytes': PER_PARTICIPANT_BYTES,
        'additional_output_cap_bytes': len(PARTICIPANTS) * PER_PARTICIPANT_BYTES,
        'children': [], 'errors': [],
        'scope': 'Existing development-only handoffs; preprocessing only, no RDM or model score',
    }

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        temporary = output_root / 'bundle.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_path)
        print(json.dumps({'completed_children': sum(item.get('complete', False)
                           for item in report['children']),
                          'elapsed_seconds': report['elapsed_seconds'],
                          'errors': report['errors']}), flush=True)

    try:
        # Metadata checks only; the unchanged child owns scientific input validation.
        for participant in PARTICIPANTS:
            handoff = read_json(source_root / participant / 'participant_handoff.json')
            sessions = handoff.get('sessions', [])
            if (handoff.get('participant') != participant
                    or handoff.get('complete_participant_handoff') is not True
                    or handoff.get('errors') != [] or len(sessions) != 30
                    or {item.get('session') for item in sessions}
                    != {f'ses-{number:02d}' for number in range(1, 31)}
                    or any(item.get('complete') is not True for item in sessions)):
                raise ValueError(f'{participant}: complete exact30-session handoff required')
            if (output_root / participant).exists() or (payload_root / participant).exists():
                raise ValueError(f'{participant}: fresh report and payload paths required')
        save()
        child_env = os.environ.copy()
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            child_env[name] = '1'
        for wave in (PARTICIPANTS[:2], PARTICIPANTS[2:]):
            active = []
            launch_error = None
            try:
                for participant in wave:
                    if INTERNAL_SECONDS - (time.monotonic() - started) < CHILD_EXTERNAL_SECONDS:
                        raise TimeoutError('No full900-second child envelope remains; stop without retry')
                    item = {'participant': participant, 'complete': False,
                            'report': str(output_root / participant / 'preprocessing.json'),
                            'started_utc': datetime.now(timezone.utc).isoformat()}
                    report['children'].append(item)
                    log = (output_root / f'{participant}.log').open('x')
                    command = [
                        'timeout', '--signal=KILL', str(CHILD_EXTERNAL_SECONDS) + 's',
                        sys.executable, 'outputs/code/preprocess_development_neural.py',
                        '--participant', participant,
                        '--participant-handoff', str(source_root / participant / 'participant_handoff.json'),
                        '--metadata-dir', 'outputs/runs/source_readiness/20261003-metadata_folds04',
                        '--payload-dir', str(payload_root / participant),
                        '--output-dir', str(output_root / participant),
                        '--max-seconds', str(CHILD_INTERNAL_SECONDS),
                    ]
                    item['command'] = command
                    try:
                        process = subprocess.Popen(command, env=child_env, stdout=log,
                                                   stderr=subprocess.STDOUT)
                    except Exception:
                        log.close()
                        raise
                    active.append((process, log, item))
                save()
            except Exception as exc:
                launch_error = exc
            finally:
                # Every launched child has its own finite timeout. Preserve the
                # other child's work even if its wave partner fails to launch.
                for process, log, item in active:
                    item['exit_code'] = process.wait()
                    log.close()
                    item['finished_utc'] = datetime.now(timezone.utc).isoformat()
                    try:
                        child = read_json(Path(item['report']))
                        item['complete'] = (
                            item['exit_code'] == 0
                            and child.get('participant') == item['participant']
                            and child.get('complete_preprocessing') is True
                            and child.get('errors') == []
                            and len(child.get('folds', [])) == 5
                            and all(fold.get('complete') is True for fold in child['folds']))
                        item['reserved_payload_upper_bytes'] = child.get('reserved_payload_upper_bytes')
                    except (OSError, ValueError) as exc:
                        item['report_error'] = str(exc)
                    save()
            if launch_error is not None:
                raise launch_error
            if any(not item['complete'] for _, _, item in active):
                raise RuntimeError('A preprocessing child failed; remaining wave is not launched')
        report['complete_remaining_participants'] = (
            len(report['children']) == len(PARTICIPANTS)
            and all(item['complete'] for item in report['children']))
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report['complete_remaining_participants'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
