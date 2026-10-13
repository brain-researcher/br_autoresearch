"""Isolated process-parallel candidate for the unchanged hierarchical fitter.

Only independent participant blocks/likelihoods run concurrently. The parent
retains the reference participant order, hyperparameter arithmetic and objective
addition order. This module is preparation, not authorization for a full fit or
general adoption of the batched urgency solver. It never submits work to Slurm.

Children are fixed spawned processes, never replaced or restarted. A monotonic
deadline bounds useful work; terminating/joining this call's children has a
separate bounded four-second cleanup allowance. Progress callbacks run only in
the parent for completed participants/sweeps and must themselves be bounded.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import collapsing_ddm_batched
import hierarchical_behavior as original
from qualify_assets import require_compute


THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
CLEANUP_SECONDS = 4.


def candidate_likelihood(theta, names, data, spec, rival):
    """Explicit candidate routing; the original fitter/likelihood stay frozen."""
    if rival == "urgency":
        return collapsing_ddm_batched.trial_likelihood(theta, names, data, spec)
    return original.trial_likelihood(theta, names, data, spec, rival)


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Parallel hierarchical fit deadline exceeded; unfinished work is technical")


def _conditional_task(payload, likelihood, subject_minimizer):
    """Literal reference conditional objective and first-finite-minimum rule."""
    began = time.monotonic()
    names, spec, data = payload["names"], payload["spec"], payload["data"]
    mean, sd, deadline = payload["mean"], payload["sd"], payload["deadline"]
    optimizer = spec["optimizer"]
    bounds = [spec["parameters"][name]["bounds"] for name in names]

    def objective(candidate):
        _check_deadline(deadline)
        try:
            nll = -likelihood(candidate, names, data, spec, payload["rival"]).sum()
        except (FloatingPointError, ValueError):
            return np.inf
        return float(nll + .5*np.sum(((candidate-mean)/sd)**2))

    attempts = []
    for start in payload["starts"]:
        _check_deadline(deadline)
        attempts.append(subject_minimizer(objective, start, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": optimizer["subject_maximum_iterations"],
                     "ftol": optimizer["subject_ftol"], "gtol": optimizer["subject_gtol"]}))
        _check_deadline(deadline)
    finite = [(index, attempt) for index, attempt in enumerate(attempts) if np.isfinite(attempt.fun)]
    if not finite:
        raise RuntimeError(f"No finite fit for {payload['participant']}")
    index, best = min(finite, key=lambda item: item[1].fun)
    return {"index": payload["index"], "participant": payload["participant"],
        "parameters": best.x, "success": bool(best.success), "message": str(best.message),
        "selected_start_index": index, "attempt_count": len(attempts),
        "elapsed_seconds": time.monotonic() - began}


def _objective_task(payload, likelihood):
    began = time.monotonic()
    _check_deadline(payload["deadline"])
    nll = -likelihood(payload["theta"], payload["names"], payload["data"],
                      payload["spec"], payload["rival"]).sum()
    _check_deadline(payload["deadline"])
    # Retain the NumPy scalar: the parent performs the same ordered += steps.
    return {"index": payload["index"], "participant": payload["participant"], "nll": nll,
            "elapsed_seconds": time.monotonic() - began}


def _worker(connection, likelihood, subject_minimizer):
    try:
        while True:
            task = connection.recv()
            if task is None:
                return
            stage, payload = task
            try:
                result = (_conditional_task(payload, likelihood, subject_minimizer) if stage == "conditional"
                          else _objective_task(payload, likelihood))
            except BaseException as error:
                connection.send((False, {"type": type(error).__name__, "message": str(error),
                    "participant": payload["participant"], "stage": stage}))
                return
            connection.send((True, result))
    except (EOFError, BrokenPipeError):
        return
    finally:
        connection.close()


def _shutdown(processes, connections):
    """Stop only children owned by this call, with one shared finite allowance."""
    for connection in connections:
        connection.close()
    processes = [process for process in processes if process.pid is not None]
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + CLEANUP_SECONDS / 2
    for process in processes:
        process.join(max(0., deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + CLEANUP_SECONDS / 2
    for process in processes:
        process.join(max(0., deadline - time.monotonic()))
    alive = [process.pid for process in processes if process.is_alive()]
    if alive:
        raise RuntimeError(f"Parallel worker cleanup did not finish for owned PIDs {alive}")


def _public_result(result):
    return {key: value.tolist() if isinstance(value, np.ndarray) else
            value.item() if isinstance(value, np.generic) else value for key, value in result.items()}


def _run_tasks(stage, tasks, processes, connections, deadline, state, progress):
    """One outstanding task per worker; never respawn after a worker failure."""
    state.update(stage=stage, completed_result_stage=stage, completed_results={},
                 pending_participants=[task["participant"] for task in tasks])
    todo = iter(tasks)
    active, results = {}, {}

    def dispatch(index):
        try:
            task = next(todo)
        except StopIteration:
            return
        _check_deadline(deadline)
        connections[index].send((stage, task))
        active[index] = task

    for index in range(len(processes)):
        dispatch(index)
    while active:
        _check_deadline(deadline)
        handles = [connections[index] for index in active] + [processes[index].sentinel for index in active]
        ready = wait(handles, timeout=max(0., deadline - time.monotonic()))
        _check_deadline(deadline)
        for index in list(active):
            connection, process = connections[index], processes[index]
            if connection not in ready:
                if process.sentinel in ready:
                    raise RuntimeError(f"Parallel worker exited before returning {active[index]['participant']}")
                continue
            try:
                success, result = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"Parallel worker lost its result for {active[index]['participant']}") from error
            if not success:
                error = RuntimeError(f"Parallel {result['stage']} failure for {result['participant']}: "
                                     f"{result['type']}: {result['message']}")
                error.worker_error = result
                raise error
            expected = active.pop(index)
            if result["index"] != expected["index"] or result["participant"] != expected["participant"]:
                raise RuntimeError("Parallel worker result identity changed")
            results[result["index"]] = result
            public = _public_result(result)
            state["completed_results"][result["participant"]] = public
            state["pending_participants"].remove(result["participant"])
            if progress is not None:
                progress({"stage": stage, "sweep": state["sweep"], "participant": result["participant"],
                    "completed_participants": len(results), "total_participants": len(tasks), "result": public})
            dispatch(index)
    return [results[index] for index in range(len(tasks))]


def fit_partially_pooled_parallel(training, model, rival, spec, *, workers, max_seconds,
                                 likelihood=None, progress=None, subject_minimizer=None):
    """Original fit schema, with bounded parallel participant computation.

    Testing may inject top-level picklable likelihood(theta,names,data,spec,rival)
    and subject_minimizer(objective,start,method=...,bounds=...,options=...).
    No reduced cohort is selected here; a scientific runner must validate its
    declared cohort before calling, as for the original generic fitter.

    An exception carries ``parallel_progress`` containing completed work and
    unresolved participants, rather than returning an apparently valid fit.
    A callback must not mutate supplied records or block indefinitely.
    """
    require_compute()
    capacity = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if (not capacity.isdecimal() or type(workers) is not int or not 1 <= workers <= int(capacity)
            or any(os.environ.get(name) != "1" for name in THREAD_ENV)):
        raise ValueError("Workers must fit actual allocated CPUs and all six numerical thread limits must be1")
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Require an explicit positive finite whole-fit work limit")
    started = time.monotonic()
    deadline = started + max_seconds
    likelihood = candidate_likelihood if likelihood is None else likelihood
    subject_minimizer = minimize if subject_minimizer is None else subject_minimizer
    processes, connections = [], []
    state = {"stage": "initialization", "sweep": 0, "completed_result_stage": None, "completed_results": {},
             "pending_participants": sorted(training), "objective_history": [], "worker_pids": []}
    caught = None
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise RuntimeError(f"Parallel hierarchical fit interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if model in {"MZC", "MVZC"} and not spec.get("joint_z_c_recovery_passed", False):
            raise ValueError("Observed joint z/c models require their recovery result first")
        names = original.parameter_names(model, rival)
        priors = [spec["parameters"][name] for name in names]
        centers = np.array([p["population_center"] for p in priors])
        pop_sd = np.array([p["population_scale"] for p in priors])
        version_sd = np.array([p["version_mean_scale"] for p in priors])
        initial_sd = np.array([p["individual_scale_center"] for p in priors])
        logsd_sd = np.array([p["individual_log_scale_sd"] for p in priors])
        bounds = [p["bounds"] for p in priors]
        if np.any(pop_sd <= 0) or np.any(version_sd <= 0) or np.any(initial_sd <= 0) or np.any(logsd_sd <= 0):
            raise ValueError("All declared prior scales must be positive")
        participants = sorted(training)
        version = np.array([0 if training[p]["version"] == "EI" else 1 for p in participants])
        if set(version) != {0, 1}:
            raise ValueError("Both task versions must be present")
        arrays = [original.make_data(training[p]["rows"]) for p in participants]
        theta = np.tile(centers, (len(participants), 1))
        global_mean = centers.copy()
        version_mean = np.tile(centers, (2, 1))
        logsd = np.tile(np.log(initial_sd), (2, 1))
        optimizer = spec["optimizer"]
        objectives, failures = [], []
        converged = False
        context = mp.get_context("spawn")
        for _ in range(workers):
            _check_deadline(deadline)
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, likelihood, subject_minimizer), daemon=True)
            connections.append(parent)
            processes.append(process)
            try:
                process.start()
            finally:
                child.close()
            state["worker_pids"].append(process.pid)
        for sweep in range(optimizer["maximum_sweeps"]):
            _check_deadline(deadline)
            state["sweep"] = sweep + 1
            failures, tasks = [], []
            for index, (participant, data) in enumerate(zip(participants, arrays)):
                group = version[index]
                mean, sd = version_mean[group], np.exp(logsd[group])
                starts = [theta[index]]
                if sweep == 0:
                    for offset in optimizer["initial_standardized_offsets"]:
                        starts.append(np.clip(centers + np.asarray(offset[:len(names)])*initial_sd,
                                              [b[0] for b in bounds], [b[1] for b in bounds]))
                tasks.append({"index": index, "participant": participant, "data": data, "names": names,
                    "spec": spec, "rival": rival, "mean": mean, "sd": sd, "starts": starts, "deadline": deadline})
            results = _run_tasks("conditional", tasks, processes, connections, deadline, state, progress)
            for index, result in enumerate(results):
                theta[index] = result["parameters"]
                if not result["success"]:
                    failures.append({"participant": participants[index], "message": result["message"]})
            state["stage"] = "hyperparameters"
            for group in [0, 1]:
                values = theta[version == group]
                variance = np.exp(2*logsd[group])
                version_mean[group] = ((values.sum(axis=0)/variance + global_mean/version_sd**2)
                                       /(len(values)/variance+1/version_sd**2))
            global_mean = ((version_mean.sum(axis=0)/version_sd**2+centers/pop_sd**2)
                           /(2/version_sd**2+1/pop_sd**2))
            for group in [0, 1]:
                values = theta[version == group]
                sumsquares = ((values-version_mean[group])**2).sum(axis=0)
                for j, prior in enumerate(priors):
                    def scale_objective(log_scale):
                        _check_deadline(deadline)
                        return (len(values)*log_scale+.5*sumsquares[j]*np.exp(-2*log_scale)
                                +.5*((log_scale-np.log(initial_sd[j]))/logsd_sd[j])**2)
                    result = minimize_scalar(scale_objective, method="bounded",
                        bounds=prior["individual_log_scale_bounds"],
                        options={"xatol": optimizer["scale_xatol"]})
                    if not result.success:
                        raise RuntimeError("Hyperparameter scale fit failed")
                    logsd[group, j] = result.x
            tasks = [{"index": index, "participant": participant, "theta": theta[index], "names": names,
                      "data": data, "spec": spec, "rival": rival, "deadline": deadline}
                     for index, (participant, data) in enumerate(zip(participants, arrays))]
            results = _run_tasks("objective", tasks, processes, connections, deadline, state, progress)
            objective_total = 0.
            for i, result in enumerate(results):
                g = version[i]
                objective_total += result["nll"]
                objective_total += np.sum(logsd[g]+.5*((theta[i]-version_mean[g])/np.exp(logsd[g]))**2)
            objective_total += .5*np.sum(((global_mean-centers)/pop_sd)**2)
            objective_total += .5*np.sum(((version_mean-global_mean)/version_sd)**2)
            objective_total += .5*np.sum(((logsd-np.log(initial_sd))/logsd_sd)**2)
            objectives.append(float(objective_total))
            state["objective_history"] = list(objectives)
            if len(objectives) > 1:
                delta = abs(objectives[-1]-objectives[-2])
                if delta <= optimizer["joint_relative_tolerance"]*(1+abs(objectives[-2])) and not failures:
                    converged = True
            if progress is not None:
                progress({"stage": "sweep", "sweep": sweep + 1, "objective": float(objective_total),
                          "converged": converged, "subject_optimizer_failures": list(failures)})
            _check_deadline(deadline)
            if converged:
                break
        state["stage"] = "finalization"
        support_contacts = []
        for index, participant in enumerate(participants):
            for j, (name, (low, high)) in enumerate(zip(names, bounds)):
                tolerance = 1e-6*(1+high-low)
                value = float(theta[index, j])
                if value-low <= tolerance or high-value <= tolerance:
                    support_contacts.append({"level": "participant", "participant": participant,
                        "parameter": name, "value": value, "bounds": [low, high]})
        for group, label in enumerate(["EI", "ER"]):
            for j, prior in enumerate(priors):
                low, high = prior["individual_log_scale_bounds"]
                tolerance = 1e-6*(1+high-low)
                value = float(logsd[group, j])
                if value-low <= tolerance or high-value <= tolerance:
                    support_contacts.append({"level": "version_log_standard_deviation",
                        "version": label, "parameter": names[j], "value": value, "bounds": [low, high]})
        _check_deadline(deadline)
        return {"model": model, "rival": rival, "parameter_names": names,
                "inference": "partially_pooled_joint_mode_plugin_predictive",
                "participants": participants, "parameters": theta.tolist(),
                "global_mean": global_mean.tolist(), "version_means": version_mean.tolist(),
                "version_standard_deviations": np.exp(logsd).tolist(),
                "version_order": ["EI", "ER"], "objective_history": objectives,
                "converged": converged, "subject_optimizer_failures": failures,
                "support_contacts": support_contacts,
                "group_interval_coverage_qualified": False,
                "individual_rank_recovery_qualified": False}
    except BaseException as error:
        caught = error
        raise
    finally:
        try:
            # A second TERM must not interrupt the bounded cleanup of children;
            # the caller's original handler is restored below on every exit.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            _shutdown(processes, connections)
            state["workers_cleaned_up"] = True
        except BaseException as cleanup_error:
            state["workers_cleaned_up"] = False
            state["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            if caught is None:
                caught = cleanup_error
                raise
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            state["elapsed_seconds"] = time.monotonic() - started
            if caught is not None:
                caught.parallel_progress = state
