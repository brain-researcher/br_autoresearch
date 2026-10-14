"""Unused completed-scalar adapter for the original EP18 ridge solver.

The caller authenticates original participant/recipe/split/model/penalty
identity in ``binding`` and supplies the same prepared arrays on continuation.
This component neither reads EEG nor authenticates the caller's data delivery.
It retains one private atomic NPZ per penalty, with all 61 columns in their
original order. Only completed zero-start LSMR calls are reusable. There is no
intermediate-recurrence resume, scoring, tuning, reduced grid or release here.

Four optional fork workers share prepared state by copy-on-write. No state is
pickled into spawned workers, worker is replaced or failed solve retried.
The coordinator owns the allocation, aggregate-memory check and outer timeout.
``progress`` exposes operational counts/times/solver status, never coefficients.
Existing numerical modules and production entrypoints are not modified.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import signal
import tempfile
import threading
import time
import zipfile

import numpy as np

from continuous_sequence_model import JointRidgeFit, JointSequenceDesign, fit_joint_ridge
from feature_extractors import require_compute


THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
SCHEMA = 'ep18_completed_scalar_lsmr_v1'
STOPS = (0, 1, 2, 4, 5)
ARRAY_NAMES = {'metadata', 'coefficients', 'completed', 'stop_codes', 'iterations',
               'elapsed_seconds', 'worker_peak_rss_bytes', 'penalties'}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def _view(values, shape, name):
    _require(isinstance(values, np.ndarray) and values.dtype == np.float64
             and values.shape == shape and np.isfinite(values).all(), name)
    result = values.view()
    result.setflags(write=False)
    return result


def _metadata(design, limits, binding):
    _require(isinstance(binding, dict) and binding, 'explicit_authenticated_fit_binding_required')
    copied = json.loads(_encoded(binding))
    _require(len(_encoded(copied)) <= 32768, 'bounded_fit_binding_required')
    solver = {name: getattr(limits, name) for name in ('atol', 'btol', 'conlim', 'maxiter')}
    _require(all(type(solver[name]) in (int, float) and math.isfinite(solver[name])
                 and solver[name] > 0 for name in ('atol', 'btol', 'conlim'))
             and solver['conlim'] > 1 and type(solver['maxiter']) is int
             and solver['maxiter'] > 0, 'original_explicit_solver_limits_required')
    return {'schema': SCHEMA, 'binding': copied, 'design_shape': list(design.shape),
            'sequence_ids': [part.sequence_id for part in design.sequences],
            'sequence_rows': [part.shape[0] for part in design.sequences],
            'solver': solver, 'contrasts': 61, 'initial_guess': 'original_zero_start'}


def _empty(columns, penalties):
    return {'coefficients': np.zeros((columns, 61), dtype=np.float64),
            'completed': np.zeros(61, dtype=np.bool_),
            'stop_codes': np.full(61, -1, dtype=np.int64),
            'iterations': np.full(61, -1, dtype=np.int64),
            'elapsed_seconds': np.zeros(61, dtype=np.float64),
            'worker_peak_rss_bytes': np.zeros(61, dtype=np.int64),
            'penalties': penalties.copy()}


def _load(path, metadata, penalties, cap):
    if not path.exists():
        return _empty(len(penalties), penalties)
    _require(path.is_file() and path.stat().st_size <= cap, 'bounded_checkpoint_required')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        _require(len(members) == len(ARRAY_NAMES)
                 and {item.filename for item in members} == {name+'.npy' for name in ARRAY_NAMES}
                 and all(item.compress_type == zipfile.ZIP_STORED for item in members)
                 and sum(item.file_size for item in members) <= cap,
                 'original_uncompressed_checkpoint_members_required')
    with np.load(path, allow_pickle=False) as archive:
        raw = archive['metadata']
        _require(raw.dtype == np.uint8 and raw.ndim == 1 and len(raw) <= 65536,
                 'bounded_checkpoint_metadata_required')
        _require(json.loads(raw.tobytes()) == metadata, 'checkpoint_binding_or_solver_mismatch')
        arrays = {name: np.array(archive[name], copy=True) for name in ARRAY_NAMES-{'metadata'}}
    expected = _empty(len(penalties), penalties)
    for name, reference in expected.items():
        value = arrays[name]
        _require(value.shape == reference.shape and value.dtype == reference.dtype
                 and np.isfinite(value).all(), 'checkpoint_array_invalid')
    _require(np.array_equal(arrays['penalties'], penalties), 'checkpoint_penalty_mismatch')
    done = arrays['completed']
    _require(np.all(np.isin(arrays['stop_codes'][done], STOPS))
             and np.all(arrays['iterations'][done] >= 0)
             and np.all(arrays['iterations'][done] <= metadata['solver']['maxiter'])
             and np.all(arrays['elapsed_seconds'][done] >= 0)
             and np.all(arrays['worker_peak_rss_bytes'][done] > 0),
             'completed_scalar_diagnostics_invalid')
    _require(np.all(arrays['coefficients'][:, ~done] == 0)
             and np.all(arrays['stop_codes'][~done] == -1)
             and np.all(arrays['iterations'][~done] == -1)
             and np.all(arrays['elapsed_seconds'][~done] == 0)
             and np.all(arrays['worker_peak_rss_bytes'][~done] == 0),
             'unfinished_scalar_cannot_be_reused')
    return arrays


def _save(path, metadata, arrays, cap):
    raw = _encoded(metadata)
    _require(len(raw) <= 65536 and sum(value.nbytes for value in arrays.values())
             + len(raw) + 16384 <= cap, 'checkpoint_record_cap_exceeded')
    # A unique pending file lets a later explicit continuation reuse the last
    # atomic checkpoint without deleting an interrupted predecessor's partial.
    with tempfile.NamedTemporaryFile(mode='wb', prefix=path.name+'.pending-',
                                     dir=path.parent, delete=False) as target:
        temporary = Path(target.name)
        np.savez(target, metadata=np.frombuffer(raw, dtype=np.uint8), **arrays)
        target.flush()
        os.fsync(target.fileno())
    _require(temporary.stat().st_size <= cap, 'checkpoint_record_cap_exceeded')
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _solve(column, design, targets, weights, penalties, solver):
    started = time.monotonic()
    fitted = fit_joint_ridge(design, targets[:, column], sample_weights=weights,
                            penalties=penalties, **solver)
    return (column, fitted.coefficients, fitted.stop_codes[0], fitted.iterations[0],
            time.monotonic()-started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def _worker(connection, design, targets, weights, penalties, solver):
    # Parent budget handlers must not prevent bounded child termination.
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        while True:
            column = connection.recv()
            if column is None:
                return
            try:
                answer = _solve(column, design, targets, weights, penalties, solver)
            except BaseException as error:
                connection.send(('error', column, type(error).__name__))
                return
            connection.send(('complete', answer))
    finally:
        connection.close()


def _finish_workers(children, connections):
    for connection in connections:
        connection.close()
    for process in children:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic()+5
    for process in children:
        process.join(max(0., deadline-time.monotonic()))
    for process in children:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic()+1
    for process in children:
        process.join(max(0., deadline-time.monotonic()))
    # A bounded join is not proof of exit (e.g. uninterruptible filesystem I/O).
    # Raising here preserves the caller's tracked PIDs: workers_finished is
    # emitted only after this function returns successfully. The caller must
    # stop this job rather than start another penalty beside a surviving child.
    _require(not any(process.is_alive() for process in children),
             'scalar_worker_cleanup_incomplete_stop_job')


def fit_checkpointed(design, targets, weights, penalties, limits, check, *,
                     checkpoint, binding, workers, record_byte_cap, progress):
    """Return all61 original columns; exceptions leave only completed cells.

    Compatible with ``folds._fit``'s six positional arguments. The caller
    selects a different private checkpoint/binding for each of all13 penalties.
    ``check()`` must enforce the actual common deadline and aggregate process
    memory. The workers_started event exposes child PIDs before work begins.
    Each completion event follows durable publication of that scalar result.
    No partial JointRidgeFit, score or scientific-release result is returned.
    """
    require_compute()
    _require(all(os.environ.get(name) == '1' for name in THREAD_ENV), 'one_BLAS_thread_per_worker_required')
    _require(type(workers) is int and 1 <= workers <= 4
             and workers <= int(os.environ.get('SLURM_CPUS_PER_TASK', '0')),
             'workers_must_fit_coordinated_allocation')
    _require(threading.current_thread() is threading.main_thread(), 'main_thread_fork_owner_required')
    _require(isinstance(design, JointSequenceDesign), 'original_joint_sequence_design_required')
    _require(type(record_byte_cap) is int and record_byte_cap > 0, 'explicit_checkpoint_cap_required')
    path = Path(checkpoint)
    _require(path.is_absolute() and '..' not in path.parts and path.parent.is_dir()
             and path.suffix == '.npz', 'explicit_private_checkpoint_path_required')
    targets = _view(targets, (design.shape[0], 61), 'complete_original_61D_targets_required')
    weights = _view(weights, (design.shape[0],), 'original_finite_weights_required')
    penalties = _view(penalties, (design.shape[1],), 'original_finite_penalties_required')
    _require(np.all(weights >= 0) and np.any(weights > 0) and np.all(penalties >= 0),
             'original_nonnegative_weight_and_penalty_rules_required')
    metadata = _metadata(design, limits, binding)
    arrays = _load(path, metadata, penalties, record_byte_cap)
    _require(sum(value.nbytes for value in arrays.values())+len(_encoded(metadata))+16384
             <= record_byte_cap, 'checkpoint_record_cap_exceeded')
    pending = [column for column in range(61) if not arrays['completed'][column]]
    progress({'event':'checkpoint_loaded', 'completed_scalars':61-len(pending), 'required_scalars':61,
              'reused_scalars':61-len(pending)})
    check()

    def commit(answer):
        column, beta, stop, count, elapsed, rss = answer
        _require(type(column) is int and 0 <= column < 61 and not arrays['completed'][column]
                 and isinstance(beta, np.ndarray) and beta.dtype == np.float64
                 and beta.shape == (design.shape[1],) and np.isfinite(beta).all()
                 and type(stop) is int and stop in STOPS and type(count) is int
                 and 0 <= count <= metadata['solver']['maxiter']
                 and math.isfinite(elapsed) and elapsed >= 0 and type(rss) is int and rss > 0,
                 'completed_original_scalar_result_required')
        arrays['coefficients'][:, column] = beta
        arrays['stop_codes'][column], arrays['iterations'][column] = stop, count
        arrays['elapsed_seconds'][column], arrays['worker_peak_rss_bytes'][column] = elapsed, rss
        arrays['completed'][column] = True
        _save(path, metadata, arrays, record_byte_cap)
        progress({'event':'scalar_completed', 'column':column,
                  'completed_scalars':int(arrays['completed'].sum()), 'required_scalars':61,
                  'stop_code':stop, 'iterations':count, 'elapsed_seconds':elapsed,
                  'worker_peak_rss_bytes':rss})

    if workers == 1:
        for column in pending:
            check()
            commit(_solve(column, design, targets, weights, penalties, metadata['solver']))
    elif pending:
        context = mp.get_context('fork')
        children, connections, active = [], [], {}
        todo = iter(pending)
        try:
            for _ in range(min(workers, len(pending))):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker,
                    args=(child, design, targets, weights, penalties, metadata['solver']), daemon=True)
                process.start()
                child.close()
                children.append(process)
                connections.append(parent)
            progress({'event':'workers_started', 'worker_pids':[item.pid for item in children],
                      'workers':len(children), 'completed_scalars':int(arrays['completed'].sum()),
                      'required_scalars':61})
            check()
            for connection in connections:
                column = next(todo)
                connection.send(column)
                active[connection] = column
            while active:
                check()
                for connection in wait(list(active), timeout=1.):
                    message = connection.recv()
                    _require(message[0] == 'complete', 'scalar_worker_failed_no_retry')
                    answer = message[1]
                    _require(answer[0] == active[connection], 'scalar_worker_column_mismatch')
                    commit(answer)
                    del active[connection]
                    column = next(todo, None)
                    if column is not None:
                        check()
                        connection.send(column)
                        active[connection] = column
                    # Keep an idle worker blocked in recv until the common
                    # cleanup. Its PID remains live for aggregate PSS checks
                    # while other original scalar fits are still running.
                for process, connection in zip(children, connections):
                    _require(connection not in active or process.is_alive(), 'scalar_worker_exited_no_retry')
        finally:
            _finish_workers(children, connections)
            progress({'event':'workers_finished', 'worker_pids':[], 'workers':0,
                      'completed_scalars':int(arrays['completed'].sum()), 'required_scalars':61})
    check()
    _require(bool(np.all(arrays['completed'])), 'all61_original_scalars_required')
    coefficients = arrays['coefficients']
    coefficients.setflags(write=False)
    return JointRidgeFit(coefficients, tuple(int(value) for value in arrays['stop_codes']),
                         tuple(int(value) for value in arrays['iterations']))
