#!/usr/bin/env python3
"""Explicit full EP17 fitting entrypoint; prospective, never an access grant.

The coordinator must first admit full development access, all mandatory
pre-score qualifications, measured costs and Slurm resources. Required options
choose a completed cost attempt, feature source, fresh fit attempt, workers,
absolute deadline and every byte/RSS cap. No resource defaults, submission,
automatic retry, endpoint evaluation or audit opening occurs here. The report
is operational; objectives, predictions and coefficients stay in one private
complete checkpoint published by the existing checkpoint writer.
"""
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
from threading import RLock
import time

EPISODE = Path(__file__).resolve().parents[2]
REPORT_ROOT = EPISODE / "outputs/runs/nested_prediction"
PRIVATE_ROOT = EPISODE / "outputs/cache/private/nested_prediction"
COST_ROOT = EPISODE / "outputs/runs/prediction_cost"


class FitRunRefusal(ValueError):
    pass


def require(condition):
    if not condition:
        raise FitRunRefusal()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _event_description(event):
    """Existing public operation coordinates, never array values or scores."""
    return {name: getattr(event, name) for name in
        ("operation", "when", "model_id", "training_folds", "layer", "rank",
         "penalty", "group", "stage", "shape") if hasattr(event, name)}


def _failure_category(error):
    """Preserve only the numerical cores' literal refusal codes, not messages."""
    kind = type(error)
    qualified = (kind.__module__, kind.__name__)
    coded = {
        ("nested_prediction", "NestedPredictionError"),
        ("parallel_nested_prediction", "ParallelPredictionError"),
        ("prediction_core", "PredictionError"),
        ("calibration_core", "CalibrationError"),
        ("released_development_inputs", "DevelopmentInputError"),
        ("predictor_checkpoint", "CheckpointError"),
    }
    code = error.args[0] if qualified in coded and len(error.args) == 1 else None
    if not (isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,159}", code)):
        code = None
    return dict(exception_class=".".join(qualified), numerical_failure_code=code)


def _write(stream, report, cap):
    encoded = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    require(len(encoded) <= cap)
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _validate(args):
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname())
    require(re.fullmatch(r"full[0-9]{3,6}", args.feature_attempt) is not None
            and re.fullmatch(r"cost[0-9]{3,6}", args.cost_attempt) is not None
            and re.fullmatch(r"fit[0-9]{3,6}", args.attempt) is not None)
    cpus = os.environ.get("SLURM_CPUS_PER_TASK", "")
    require(cpus.isdecimal() and type(args.workers) is int and 1 <= args.workers <= min(4, int(cpus)))
    require(all(os.environ.get(name) == "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
            and all(os.environ.get(name, "1") == "1" for name in ("VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")))
    require(math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time())
    require(all(type(value) is int and value > 0 for value in (args.rss_limit_kib, args.metadata_byte_cap,
        args.feature_byte_cap, args.response_byte_cap, args.checkpoint_byte_cap, args.report_byte_cap)))


def _completed_cost(args, inputs, cost, check):
    value = inputs.existing._json(COST_ROOT / args.cost_attempt / "cost_report.json", args.metadata_byte_cap, check)
    measured = value.get("cost_report")
    require(value.get("analysis_id") == "prediction_qualification" and value.get("stage") == "prediction_cost"
            and value.get("attempt") == args.cost_attempt and value.get("feature_attempt") == args.feature_attempt
            and value.get("status") == "completed" and isinstance(measured, dict)
            and measured.get("status") == "complete" and measured.get("failure_code") is None)
    require(all(measured.get("completed_counts", {}).get(name) == count for name, count in cost.REQUIRED_COUNTS.items()))


def _run(args):
    import private_stimulus_feature_pilot as storage
    from run_private_development_features import Budget

    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    directory, private = REPORT_ROOT / args.attempt, PRIVATE_ROOT / args.attempt
    require(all(not path.exists() and not path.is_symlink() for path in (directory, private)))
    storage._mkdir_private(directory, exclusive=True)
    storage._mkdir_private(private, exclusive=True)
    destination = private / "complete_prediction.zip"
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="prediction_qualification", stage="complete_nested_prediction", attempt=args.attempt,
        feature_attempt=args.feature_attempt, cost_attempt=args.cost_attempt, status="started", started_utc=stamp(),
        job_id=os.environ["SLURM_JOB_ID"], step_id=os.environ.get("SLURM_STEP_ID"), host=socket.gethostname(),
        workers=args.workers, numerical_threads_per_worker=1, model_count=4,
        response_mean_bytes=None, loader_dimensions=None, neural_response_read_attempted=False,
        private_inner_scoring_attempted=False, checkpoint_published=False,
        checkpoint_path=str(destination), audit_access=False, endpoint_evaluation=False,
        caps=dict(deadline_epoch=args.deadline_epoch, process_peak_rss_kib=args.rss_limit_kib,
                  metadata_bytes=args.metadata_byte_cap, feature_bytes=args.feature_byte_cap,
                  response_mean_bytes=args.response_byte_cap, checkpoint_bytes=args.checkpoint_byte_cap,
                  report_bytes=args.report_byte_cap))
    phase, published, last_event = "scientific_imports", None, {}
    lock = RLock()
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def stop(_signal, _frame):
        raise FitRunRefusal()

    with storage._exclusive_file(directory / "fit_report.json") as stream:
        _write(stream, report, args.report_byte_cap)
        try:
            for sig in saved:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            import complete_development_inputs as inputs
            import development_prediction_cost as cost
            import parallel_nested_prediction as parallel
            import predictor_checkpoint as checkpoint
            budget()
            inputs.existing.feature_io.delivery.features.check_runtime()

            def check(event=None):
                nonlocal last_event
                with lock:
                    if event is not None:
                        last_event = _event_description(event)
                    budget(event)
                    operation, when = getattr(event, "operation", None), getattr(event, "when", "before")
                    dirty = False
                    if operation == "plan_complete_response_means":
                        report["response_mean_bytes"] = event.byte_count
                        _write(stream, report, args.report_byte_cap)
                        require(0 < event.byte_count <= args.response_byte_cap)
                    elif operation == "read_stripe" and not report["neural_response_read_attempted"]:
                        report["neural_response_read_attempted"] = True
                        report["first_response_read_attempt_utc"], dirty = stamp(), True
                    elif operation == "all_features_feasible" and when == "after":
                        report["all_352_PCA_fits_complete_utc"], dirty = stamp(), True
                    elif operation == "inner_error_sums" and not report["private_inner_scoring_attempted"]:
                        report["private_inner_scoring_attempted"] = True
                        report["first_private_inner_scoring_attempt_utc"], dirty = stamp(), True
                    if dirty:
                        _write(stream, report, args.report_byte_cap)

            phase = "selected_cost_binding"
            _completed_cost(args, inputs, cost, check)
            phase = "complete_development_loading"
            source = inputs.load_complete_development_inputs(feature_attempt=args.feature_attempt,
                feature_byte_cap=args.feature_byte_cap, metadata_byte_cap=args.metadata_byte_cap,
                response_byte_cap=args.response_byte_cap, resource_check=check)
            require(source.complete_development_response_release is True)
            report["loader_dimensions"] = dict(source.operational_dimensions)
            report["input_loading_seconds"] = time.monotonic() - started
            _write(stream, report, args.report_byte_cap)
            phase = "complete_nested_fitting"
            result = parallel.run_parallel_nested_prediction(source.design, source.features, source.responses,
                source.frozen_columns, workers=args.workers, resource_check=check)
            report["complete_fitting_seconds_including_input_loading"] = time.monotonic() - started
            del source
            phase = "checkpoint_publication"
            published = checkpoint.save_prediction_checkpoint(result, destination,
                max_bytes=args.checkpoint_byte_cap, resource_check=check)
            report.update(status="completed", checkpoint_published=True,
                          partial_cleanup_complete=published.partial_cleanup_complete)
        except BaseException as error:
            report["status"], report["error_code"] = "stopped", phase + "_failed"
            report.update(_failure_category(error))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=stamp(), total_pipeline_wall_seconds=time.monotonic() - started,
                total_pipeline_process_cpu_seconds=time.process_time() - cpu_started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                completed_operation_counts=dict(budget.completed), last_operation=budget.last_operation,
                last_resource_event=last_event,
                checkpoint_publisher_returned=published is not None,
                checkpoint_path_present=destination.exists())
            # A signal during publication can leave a committed checkpoint.
            # Preserve it and require reconciliation; never refit/overwrite.
            _write(stream, report, args.report_byte_cap)
    return 0 if report["status"] == "completed" and report["checkpoint_published"] and published is not None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("feature-attempt", "cost-attempt", "attempt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    for name in ("workers", "rss-limit-kib", "metadata-byte-cap", "feature-byte-cap", "response-byte-cap",
                 "checkpoint-byte-cap", "report-byte-cap"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args(argv)
    try:
        _validate(args)
        code = _run(args)
    except BaseException:
        code = 1
    print(json.dumps({"status": "completed" if code == 0 else "stopped", "attempt": args.attempt}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
