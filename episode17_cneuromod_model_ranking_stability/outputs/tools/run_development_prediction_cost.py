#!/usr/bin/env python3
"""Explicit, prospective EP17 prediction-cost CLI; no automatic reader release.

The coordinator must log the exact procedural builder handoff after the full
feature bank completes, select real Slurm resources and an external timeout,
then supply every resource/output choice below. A 4 GiB parent is not assumed
adequate. The absolute deadline, alarm and observed RSS callbacks cover imports,
input loading and the unchanged complete cost probe. Slurm/outer timeout bound
native kernels. No full fitting, validation score, artifact release or retry.
"""
from collections.abc import Mapping
from dataclasses import fields
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
import sys
import time

EPISODE = Path(__file__).resolve().parents[2]
REPORT_ROOT = EPISODE / "outputs/runs/prediction_cost"


class CostRunRefusal(ValueError):
    pass


def require(condition):
    if not condition:
        raise CostRunRefusal()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _write(stream, report, cap):
    encoded = (json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8")
    require(len(encoded) <= cap)
    stream.seek(0)
    stream.write(encoded)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def _cost_fields(value):
    # MappingProxyType cannot be deep-copied by dataclasses.asdict. The cost
    # report already contains only scalar operational fields and scalar maps.
    return {field.name: dict(item) if isinstance(item := getattr(value, field.name), Mapping) else item
            for field in fields(value)}


def _validate(args):
    require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname())
    require(re.fullmatch(r"full[0-9]{3,6}", args.feature_attempt) is not None
            and re.fullmatch(r"cost[0-9]{3,6}", args.attempt) is not None)
    require(math.isfinite(args.deadline_epoch) and args.deadline_epoch > time.time())
    require(all(type(value) is int and value > 0 for value in
                (args.rss_limit_kib, args.metadata_byte_cap, args.feature_byte_cap,
                 args.response_byte_cap, args.report_byte_cap)))


def _run(args):
    # These two modules use stdlib only at import. Guard all scientific imports
    # below after the durable started report and whole-pipeline alarm exist.
    import private_stimulus_feature_pilot as storage
    from run_private_development_features import Budget

    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    directory = REPORT_ROOT / args.attempt
    require(not directory.exists() and not directory.is_symlink())
    storage._mkdir_private(directory, exclusive=True)
    path = directory / "cost_report.json"
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="prediction_qualification", stage="prediction_cost", attempt=args.attempt,
        feature_attempt=args.feature_attempt, status="started", started_utc=stamp(),
        job_id=os.environ["SLURM_JOB_ID"], step_id=os.environ.get("SLURM_STEP_ID"), host=socket.gethostname(),
        cost_report=None, loader_dimensions=None, response_header_bytes=None,
        neural_training_read_attempted=False,
        caps=dict(deadline_epoch=args.deadline_epoch, observed_peak_rss_kib=args.rss_limit_kib,
                  metadata_bytes=args.metadata_byte_cap, feature_bytes=args.feature_byte_cap,
                  response_bytes=args.response_byte_cap, report_bytes=args.report_byte_cap))
    phase = "scientific_imports"
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def stop(_signal, _frame):
        raise CostRunRefusal()

    with storage._exclusive_file(path) as stream:
        _write(stream, report, args.report_byte_cap)
        try:
            for sig in saved:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            import released_development_inputs as inputs
            import development_prediction_cost as cost
            budget()
            inputs.feature_io.delivery.features.check_runtime()

            def check(event=None):
                budget(event)
                operation = getattr(event, "operation", event)
                if operation == "plan_cost_response_headers":
                    report["response_header_bytes"] = event.byte_count
                    _write(stream, report, args.report_byte_cap)
                    require(0 < event.byte_count <= args.response_byte_cap)
                elif operation == "read_training_column" and not report["neural_training_read_attempted"]:
                    report["neural_training_read_attempted"] = True
                    report["first_training_read_attempt_utc"] = stamp()
                    _write(stream, report, args.report_byte_cap)

            phase = "loading_cost_inputs"
            source = inputs.load_development_prediction_cost_inputs(feature_attempt=args.feature_attempt,
                feature_byte_cap=args.feature_byte_cap, metadata_byte_cap=args.metadata_byte_cap,
                response_byte_cap=args.response_byte_cap, resource_check=check)
            require(source.complete_development_response_release is False)
            report["loader_dimensions"] = dict(source.operational_dimensions)
            report["input_loading_seconds"] = time.monotonic() - started
            _write(stream, report, args.report_byte_cap)
            phase = "prediction_cost"
            measured = cost.measure_development_prediction_cost(source.design, source.features,
                source.responses, source.frozen_columns, resource_check=check)
            report["cost_report"] = _cost_fields(measured)
            report["status"] = "completed" if measured.status == "complete" else "stopped"
            if measured.status != "complete":
                report["error_code"] = "prediction_cost_stopped"
            del source
        except BaseException:
            report["status"], report["error_code"] = "stopped", phase + "_failed"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=stamp(), total_pipeline_wall_seconds=time.monotonic() - started,
                total_pipeline_process_cpu_seconds=time.process_time() - cpu_started,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                last_operation=budget.last_operation)
            _write(stream, report, args.report_byte_cap)
    return 0 if (report["status"] == "completed" and report["cost_report"] is not None
                 and report["cost_report"]["status"] == "complete") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-attempt", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline-epoch", required=True, type=float)
    for name in ("rss-limit-kib", "metadata-byte-cap", "feature-byte-cap", "response-byte-cap", "report-byte-cap"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args(argv)
    try:
        _validate(args)
        result = _run(args)
    except BaseException:
        result = 1
    print(json.dumps({"status": "completed" if result == 0 else "stopped", "attempt": args.attempt}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
