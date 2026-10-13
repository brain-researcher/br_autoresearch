#!/usr/bin/env python3
"""One sequential participant handoff; exact frozen sessions and no retries."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from inspect_array_metadata import PARTICIPANTS, read_json
from prepare_development_neural_pilot import EPISODE, ROI_NAMES, SCRATCH_LIMIT


SESSIONS = tuple(f'ses-{number:02d}' for number in range(1, 31))


def session_plan(participant):
    if participant not in PARTICIPANTS:
        raise ValueError('Participant is outside the frozen cohort')
    return [(session, participant == 'sub-01' and session == 'ses-01') for session in SESSIONS]


def require_complete_session(report, participant, session, source_root, frozen_trial):
    selected = frozen_trial['role_counts']['development_unique']
    total = frozen_trial['rows']
    if (report.get('complete_neural_session_handoff') is not True
            or report.get('participant') != participant or report.get('session') != session
            or report.get('source_root') != str(source_root) or report.get('errors') != []
            or report.get('roi_names') != list(ROI_NAMES)
            or report.get('selected_trial_count') != selected
            or report.get('selected_volumes_decoded') != selected
            or report.get('total_trial_count') != total
            or report.get('source_volumes_consumed') != total
            or report.get('excluded_volumes_discarded_as_opaque_bytes') != total - selected
            or report.get('excluded_beta_volumes_numerically_decoded') != 0
            or report.get('session_standardization_reliability_RDM_or_model_fit_performed') is not False
            or report.get('raw_images_or_noise_ceiling_maps_read') is not False):
        raise ValueError('Session completion, identity, frozen role counts or numerical boundary disagree')
    union = report.get('union_voxel_count')
    reserved = report.get('planned_payload_upper_bytes')
    if (type(union) is not int or union <= 0
            or report.get('selected_beta_payload_bytes') != 4 * selected * union
            or type(reserved) is not int or not 0 < reserved <= SCRATCH_LIMIT):
        raise ValueError('Session dimensions/storage reservation disagree')
    return reserved


def remaining_budget(reservations, limit=SCRATCH_LIMIT):
    if any(type(value) is not int or value <= 0 for value in reservations):
        raise ValueError('Invalid positive storage reservation')
    remaining = limit - sum(reservations)
    if remaining <= 0:
        raise ValueError('Participant scratch cap reached; no further session is launched')
    return remaining


def artifact_presence(report):
    """Check preserved output presence/byte lengths, never load beta or ROI arrays."""
    root = Path(report['payload_dir'])
    names = ('development_betas.npy', 'voxel_indices_c.npy', 'roi_membership.npy', 'development_trials.jsonl')
    for name in names:
        path = root / name
        if report['artifacts'].get(name) != str(path) or not path.is_file():
            raise ValueError('Preserved session artifact is missing or routed elsewhere')
    payload_sizes = {'development_betas.npy': report['selected_beta_payload_bytes'],
                     'voxel_indices_c.npy': 8 * report['union_voxel_count'],
                     'roi_membership.npy': 4 * report['union_voxel_count']}
    for name, values_bytes in payload_sizes.items():
        header_bytes = (root / name).stat().st_size - values_bytes
        if not 64 <= header_bytes <= 1024 * 1024:
            raise ValueError('Preserved NPY length is inconsistent with completed dimensions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--participant-index', type=int, choices=range(5), required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    parser.add_argument('--array-metadata', type=Path, required=True)
    parser.add_argument('--roi-mapping', type=Path, required=True)
    parser.add_argument('--pilot-report', type=Path, required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=1740)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or '-ln' in socket.gethostname():
        raise RuntimeError('Coordinator-owned verified compute allocation required')
    if not 0 < args.max_seconds <= 1740:
        raise ValueError('Participant internal envelope must be positive and<=1740 seconds')
    participant = PARTICIPANTS[args.participant_index]
    report_dir = args.output_root / participant
    payload_dir = args.payload_root / participant
    scratch = os.environ.get('SCRATCH')
    if (not scratch or not payload_dir.resolve().is_relative_to(
            (Path(scratch) / 'br_autoresearch' / EPISODE).resolve())
            or report_dir.exists() or payload_dir.exists()):
        raise ValueError('Use fresh participant report/payload directories under episode scratch')
    report_dir.mkdir(parents=True)
    payload_dir.mkdir(parents=True)
    started = time.monotonic()
    report = {'analysis_id': 'source_readiness', 'participant': participant,
        'job_id': os.environ['SLURM_JOB_ID'], 'array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'hostname': socket.gethostname(), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'complete_participant_handoff': False, 'all150_handoff_claim': False,
        'internal_envelope_seconds': args.max_seconds, 'scratch_limit_bytes': SCRATCH_LIMIT,
        'source_root': str(args.source_root), 'sessions': [], 'errors': [],
        'no_retry': True, 'numerical_scope': 'Development-only original beta/mask handoff; no session standardization, reliability, RDMs or models'}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        temporary = report_dir / 'participant_handoff.inprogress.json'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_dir / 'participant_handoff.json')
        print(json.dumps({'participant': participant, 'recorded_sessions': len(report['sessions']),
                          'elapsed_seconds': report['elapsed_seconds']}), flush=True)

    reservations, reference_support = [], None
    try:
        frozen = read_json(args.metadata_dir / 'source_metadata_report.json')
        if (frozen.get('metadata_fold_freeze_complete') is not True
                or frozen.get('source_root') != str(args.source_root)):
            raise ValueError('Completed frozen source-role metadata required')
        trial_rows = [row for row in frozen['trial_files'] if row['participant'] == participant]
        if len(trial_rows) != 30 or {row['session'] for row in trial_rows} != set(SESSIONS):
            raise ValueError('Exactly30 frozen sessions are required for this participant')
        trials = {row['session']: row for row in trial_rows}
        for session, reuse in session_plan(participant):
            remaining = remaining_budget(reservations)
            item = {'session': session, 'reused_pilot': reuse, 'complete': False}
            report['sessions'].append(item)
            if reuse:
                selected_report = args.pilot_report
                item['source_beta_or_ROI_arrays_reread'] = False
            else:
                seconds_left = int(args.max_seconds - (time.monotonic() - started)) - 20
                if seconds_left <= 0:
                    raise TimeoutError('Participant envelope leaves no time for another bounded session')
                session_limit = min(540, seconds_left)
                selected_report = report_dir / session / 'neural_handoff.json'
                command = [sys.executable, str(Path(__file__).with_name('prepare_development_neural_pilot.py')),
                    '--source-root', str(args.source_root), '--metadata-dir', str(args.metadata_dir),
                    '--array-metadata', str(args.array_metadata), '--roi-mapping', str(args.roi_mapping),
                    '--payload-dir', str(payload_dir / session), '--output-dir', str(report_dir / session),
                    '--participant', participant, '--session', session,
                    '--scratch-limit-bytes', str(remaining), '--max-seconds', str(session_limit)]
                item['internal_session_seconds'] = session_limit
                item['report'] = str(selected_report)
                save()  # Preserve selected session even if Slurm must stop the process.
                completed = subprocess.run(command, check=False)
                item['exit_code'] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(f'Session {session} failed; preserved without retry')
            item['report'] = str(selected_report)
            session_report = read_json(selected_report)
            reservation = require_complete_session(session_report, participant, session, args.source_root, trials[session])
            if reservation > remaining:
                raise ValueError('Completed session exceeds remaining participant scratch reservation')
            artifact_presence(session_report)
            support = (session_report['union_voxel_count'], session_report['roi_voxel_counts'])
            if reference_support is not None and support != reference_support:
                raise ValueError('Fixed participant ROI support changed across sessions')
            reference_support = support
            reservations.append(reservation)
            item.update(complete=True, reserved_payload_bytes=reservation,
                        selected_trial_count=session_report['selected_trial_count'],
                        selected_beta_payload_bytes=session_report['selected_beta_payload_bytes'],
                        selected_roi_nonfinite_values=session_report['selected_roi_nonfinite_values'],
                        selected_roi_zero_values=session_report['selected_roi_zero_values'])
            report['reserved_payload_bytes_including_reused_pilot'] = sum(reservations)
            save()
        report['complete_participant_handoff'] = (len(report['sessions']) == 30
                                                and all(item['complete'] for item in report['sessions']))
        report['selected_trial_count'] = sum(item['selected_trial_count'] for item in report['sessions'])
    except Exception as error:
        report['errors'].append({'type': type(error).__name__, 'message': str(error)})
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return 0 if report['complete_participant_handoff'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
