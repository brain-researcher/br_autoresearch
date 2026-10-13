#!/usr/bin/env python3
"""Diagnose the fixed sub-01 initialization prefix; emit metadata-only JSON lines.

Run from EP04 in a coordinator-owned one-CPU compute step, externally bounded
by 120 seconds plus 10 seconds of termination grace. There are no source or
participant overrides. The existing loader validates source/ROI metadata and
opens beta memmaps for shape/dtype only. No beta values, fold parameters, NC,
reliability, RDMs or model scores are computed or emitted.

A single 90-second faulthandler deadline emits the interrupted Python stack
to stderr and exits. It is not a retry, scientific run or infrastructure probe.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
PREPROCESSING_REPORT = Path(
    'outputs/runs/neural_preprocessing/20261003-remaining-folds01/sub-01/preprocessing.json'
)
METADATA_DIRECTORY = Path('outputs/runs/source_readiness/20261003-metadata_folds04')
THREAD_ENV = (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    started = time.monotonic()
    cpu_started = time.process_time()
    hostname = socket.gethostname()
    job_id = os.environ.get('SLURM_JOB_ID')

    def mark(phase, **details):
        print(json.dumps({
            'phase': phase,
            'hostname': hostname,
            'job_id': job_id,
            'elapsed_seconds': time.monotonic() - started,
            'process_cpu_seconds': time.process_time() - cpu_started,
            **details,
        }, allow_nan=False), flush=True)

    mark('entry', cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
         numerical_threads={key: os.environ.get(key) for key in THREAD_ENV},
         traceback_deadline_seconds=90)
    if (not job_id or '-ln' in hostname
            or os.environ.get('SLURM_CPUS_PER_TASK') != '1'
            or any(os.environ.get(key) != '1' for key in THREAD_ENV)):
        raise RuntimeError('Use a verified one-CPU compute step with all six numerical threads set to1')
    if Path.cwd().resolve() != EPISODE:
        raise ValueError('Run only from the named EP04 episode directory')

    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=False, exit=True)
    try:
        mark('before_existing_loader_import')
        from preprocess_development_neural import load_completed_inputs

        mark('before_selected_preprocessing_metadata', path=str(PREPROCESSING_REPORT))
        prepared = json.loads(PREPROCESSING_REPORT.read_text())
        folds = prepared.get('folds', [])
        if (prepared.get('participant') != 'sub-01'
                or prepared.get('complete_preprocessing') is not True
                or prepared.get('errors') != [] or len(folds) != 5
                or {fold.get('held_fold') for fold in folds} != set(range(5))
                or not all(fold.get('complete') is True for fold in folds)):
            raise ValueError('The exact completed sub-01 five-fold preprocessing report is required')

        mark('before_exact_completed_handoff_loader',
             participant_handoff=prepared['participant_handoff'],
             metadata_directory=str(METADATA_DIRECTORY))
        images, sessions, indices, _membership = load_completed_inputs(
            'sub-01', Path(prepared['participant_handoff']), METADATA_DIRECTORY
        )
        mark('initialization_prefix_returned', images=len(images),
             sessions=len(sessions), voxels=len(indices),
             numerical_beta_rows_read=0, fold_parameters_read=False,
             training_context_computed=False)
    except Exception as exc:
        mark('initialization_prefix_failed', error=f'{type(exc).__name__}: {exc}')
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
