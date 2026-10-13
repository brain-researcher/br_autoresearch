"""One admitted first-fold category discriminator; never fit or normalize.

Run only inside the explicitly selected substantive parent: attempt001 uses
1CPU/7GiB; manually selected attempt002 uses4CPU/31GiB serially, while retaining
one numerical thread and the same300s/6.5GiB process-RSS/1MiB output limits. The
coordinator supplies its verified job ID and EndTime; this entrypoint performs
no scheduler query. The original input binder and original lock remain intact.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

BASE = Path(__file__).resolve().parents[3]
LOCK = BASE / "outputs/protocol_lock_brain_constant_discriminator_2026-10-12.json"
RUN = BASE / "outputs/runs/brain_constant_discriminator/attempt001"
ATTEMPT_LOCKS = {
    "attempt001": "outputs/protocol_lock_brain_constant_discriminator_2026-10-12.json",
    "attempt002": "outputs/protocol_lock_brain_constant_discriminator_attempt002_2026-10-12.json",
}
LIMIT = 6979321856
SCOPE = {"source_tau_train": "sub-03", "target_tau_train": "sub-01",
         "training_images": 897, "source_native_voxels": 12897,
         "target_native_voxels": 12295, "fold": 0}
RESOURCES = {"cpus": 1, "parent_memory_gib": 7, "gpus": 0,
             "numeric_threads": 1, "deadline_seconds": 300,
             "minimum_parent_remaining_seconds": 330,
             "rss_limit_bytes": LIMIT, "durable_bytes": 1048576}
ATTEMPT_RESOURCES = {
    "attempt001": dict(RESOURCES),
    "attempt002": {**RESOURCES, "cpus": 4, "parent_memory_gib": 31},
}


def require(condition, code):
    if not condition:
        raise ValueError(code)


def read_lock():
    with LOCK.open("rb") as stream:
        raw = stream.read(32769)
    require(len(raw) <= 32768, "conditional_lock_size")
    value = json.loads(raw)
    expected = {"schema_version": "ep21.conditional_brain_discriminator_lock.v1",
        "episode_id": "ep21", "state": "conditional_amendment_discriminator_only",
        "scientist_decision": "yes, adopt the sub-01 fix conditionally",
        "decision_source": "outputs/experiment_log.md#2026-10-12-1616-pacific--interindividual_converter_training--scientist-adopts-the-constant-coordinate-comparator-conditionally",
        "rule_source": "outputs/proposed_brain_normalization_followup.md",
        "inherited_lock": "outputs/protocol_lock.yaml",
        "authorized_operation": "discriminate_brain_first_fold_constants",
        "input_scope": SCOPE, "resources": RESOURCES,
        "run_directory": str(RUN.relative_to(BASE)),
        "original_validator_and_artifacts_preserved": True,
        "original_not_evaluable_disposition_preserved": True}
    for key in ("amended_implementation_authorized_before_guard", "fitting_authorized",
                "model_publication_authorized", "scoring_authorized",
                "tau_test_access_authorized", "OOD_access_authorized",
                "automatic_retry", "readout_release_authorized"):
        expected[key] = False
    require(all(value.get(key) == item for key, item in expected.items()),
            "conditional_lock_scope_changed")
    return value


class Budget:
    def __init__(self, job, end):
        host = socket.gethostname().split(".")[0]
        require(job.isdigit() and os.environ.get("SLURM_JOB_ID") == job
                and "-ln" not in host, "actual_selected_compute_parent_required")
        require(os.environ.get("SLURM_CPUS_PER_TASK") == str(RESOURCES["cpus"])
                and os.environ.get("SLURM_MEM_PER_NODE") == str(RESOURCES["parent_memory_gib"] * 1024),
                "selected_actual_parent_resources_required")
        require(math.isfinite(end) and end - time.time() >= 330,
                "parent_remaining_time_insufficient")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            require(os.environ.get(key) == "1", "one_numerical_thread_required")
        self.allocation = {"job_id": job, "hostname": host, "job_end_epoch": end,
                           "cpus": RESOURCES["cpus"], "memory_gib": RESOURCES["parent_memory_gib"]}
        self.deadline = time.monotonic() + 300
        self.profile = read_lock()
        self.check()

    def check(self):
        if time.monotonic() >= self.deadline or time.time() >= self.allocation["job_end_epoch"] - 20:
            raise TimeoutError("discriminator_deadline")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > LIMIT:
            raise MemoryError("discriminator_RSS_limit")


def resolver(context, *, operation, source, target, budget, execution_options):
    require(context is budget and isinstance(budget, Budget)
            and read_lock() == budget.profile, "conditional_admission_changed")
    require((source, target) == ("sub-03", "sub-01") and execution_options is None
            and operation in ("bind_brain_inputs", "discriminate_brain_first_fold_constants"),
            "operation_not_admitted")
    budget.check()


def discriminate(bound):
    import numpy as np
    import brain_loss as brain
    from empirical_brain_inputs import _admit
    budget = bound.budget
    resolver(bound.execution_context, operation="discriminate_brain_first_fold_constants",
             source="sub-03", target="sub-01", budget=budget, execution_options=None)
    _admit(bound.execution_context, bound.require_admitted_context, "sub-03", "sub-01",
           budget, bound.allocation_identity)
    for response, who, width in ((bound.source, "sub-03", 12897), (bound.target, "sub-01", 12295)):
        require(response.participant == who and response.role == "tau_train"
                and len(response.native_voxel_ids) == width
                and response.values.shape == (897, width), "full_original_native_scope_required")
    brain._validate_folds(bound.folds)
    x, _ = brain._aligned_values(bound.source, bound.folds, bound.source.native_voxel_ids)
    y, _ = brain._aligned_values(bound.target, bound.folds, bound.target.native_voxel_ids)
    fitting = np.ones(brain.TRAINING_IMAGES, dtype=bool)
    fitting[np.asarray(bound.folds.validation_indices[0])] = False
    budget.check()
    # Preserve the original prefix: source moments must pass before classifying
    # the known target failure. Do not normalize either fitting/validation data.
    brain._moments(x[fitting], "source_fold_0")
    values = y[fitting]
    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.mean(values, axis=0)[None, :]
        std = np.std(values, axis=0, ddof=1)[None, :]
    budget.check()
    bad_mean = ~np.isfinite(mean[0])
    bad_sd = ~np.isfinite(std[0])
    negative = np.isfinite(std[0]) & (std[0] < 0)
    zero = np.isfinite(std[0]) & (std[0] == 0)
    constant = np.isfinite(values).all(axis=0) & np.all(values == values[0], axis=0)
    exact_mean = np.isfinite(mean[0]) & (mean[0] == values[0])
    eligible = zero & constant & exact_mean
    invalid = bad_mean | bad_sd | negative | zero
    count = lambda mask: int(np.count_nonzero(mask))
    categories = {"nonfinite_means": count(bad_mean), "nonfinite_SDs": count(bad_sd),
        "negative_SDs": count(negative), "finite_zero_SDs": count(zero),
        "zero_SD_exact_constants_exact_mean": count(eligible),
        "zero_SD_exact_constants_inexact_mean": count(zero & constant & ~exact_mean),
        "zero_SD_varying_columns": count(zero & ~constant),
        "total_invalid_column_union": count(invalid)}
    budget.check()
    return {"counts": categories,
            "guard_satisfied": bool(np.any(invalid) and np.array_equal(invalid, eligible)),
            "first_fold_only": True, "all_fold_eligibility_established": False}


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(len(raw) <= 65536, "categorical_output_bound")
    # Exactly two bounded files; no arrays or directory-wide scans.
    other = RUN / ("claim.json" if name == "report.json" else "report.json")
    require(len(raw) + (other.stat().st_size if other.exists() else 0) <= 1048576,
            "durable_output_bound")
    with (RUN / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def interrupt(signum, frame):
    raise TimeoutError("discriminator_interrupted")


def run(job, end, attempt="attempt001"):
    global LOCK, RUN, RESOURCES
    require(attempt in ATTEMPT_LOCKS, "explicit_registered_attempt_required")
    LOCK = BASE / ATTEMPT_LOCKS[attempt]
    RUN = BASE / "outputs/runs/brain_constant_discriminator" / attempt
    RESOURCES = dict(ATTEMPT_RESOURCES[attempt])
    started = time.monotonic()
    budget = Budget(job, end)
    RUN.parent.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(mode=0o700)  # Existing attempts are never reused or overwritten.
    report = {"kind": "ep21_brain_constant_discriminator", "status": "incomplete",
        "source": "sub-03", "target": "sub-01", "fold": 0,
        "fits_performed": 0, "scores_produced": False, "normalization_applied": False,
        "model_published": False, "tau_test_accessed": False,
        "automatic_retry": False, "additional_allocation_charge": False}
    publish("claim.json", {"conditional_lock": str(LOCK.relative_to(BASE)),
        "allocation": budget.allocation, "resources": RESOURCES,
        "additional_allocation_charge": False})
    prior = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGALRM, signal.SIGTERM)}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, budget.deadline - time.monotonic()))
    stage = "scientific_imports"
    try:
        with open(os.devnull, "w") as quiet, redirect_stdout(quiet), redirect_stderr(quiet):
            from empirical_brain_inputs import bind_brain_inputs
            budget.check()
            stage = "original_tau_train_binding"
            bound = bind_brain_inputs(budget, require_admitted_context=resolver,
                source="sub-03", target="sub-01", budget=budget)
            stage = "first_fold_category_counts"
            report.update(discriminate(bound))
            budget.check()
            report["status"] = "completed"
    except BaseException as error:
        report.update(status="incomplete", error_type=type(error).__name__, failed_stage=stage)
        frames = []
        trace = error.__traceback__
        while trace is not None:
            frames.append({"file": Path(trace.tb_frame.f_code.co_filename).name,
                           "function": trace.tb_frame.f_code.co_name, "line": trace.tb_lineno})
            trace = trace.tb_next
        report["source_frames"] = frames[-8:]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in prior.items():
            signal.signal(sig, handler)
    report.update(elapsed_seconds=time.monotonic() - started,
                  peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    publish("report.json", report)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--parent-end-epoch", required=True, type=float)
    parser.add_argument("--attempt", choices=tuple(ATTEMPT_LOCKS), default="attempt001")
    args = parser.parse_args()
    try:
        result = run(args.parent_job_id, args.parent_end_epoch, args.attempt)
    except BaseException as error:
        print(json.dumps({"status": "entrypoint_refused", "error_type": type(error).__name__}))
        result = 2
    raise SystemExit(result)
