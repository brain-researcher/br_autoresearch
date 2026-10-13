#!/usr/bin/env python3
"""One explicit EP17 fit001 response-preparation diagnosis, never a retry.

Reopen only the existing released full001 development inputs, then run the
existing _row_indices/_prepare_responses path without PCA, model fitting,
candidate errors, endpoint scoring or audit access. Preserve fit001. The new
exclusive operational report records categorical refusal and metadata location
only, never response magnitudes, voxel identities, model objectives or scores.
Coordinator admission: one CPU, 7 GiB, at most15min, with a6GiB observed RSS
guard and an external timeout. No successor/automatic retry is available.
"""
import argparse
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time
import json

import run_complete_development_prediction as fitting


EPISODE = Path(__file__).resolve().parents[2]
ATTEMPT = "fit001_diagnostic001"
OUTPUT = EPISODE / "outputs/runs/nested_prediction" / ATTEMPT


def _run(args):
    import private_stimulus_feature_pilot as storage
    from run_private_development_features import Budget
    require = fitting.require
    budget = Budget(args.deadline_epoch, args.rss_limit_kib)
    budget()
    require(not OUTPUT.exists() and not OUTPUT.is_symlink())
    storage._mkdir_private(OUTPUT, exclusive=True)
    started, cpu_started = time.monotonic(), time.process_time()
    report = dict(analysis_id="prediction_qualification", stage="fit001_response_variance_diagnosis",
        attempt=ATTEMPT, original_attempt="fit001", feature_attempt="full001", status="started",
        started_utc=fitting.stamp(), job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
        model_fitting=False, PCA_fitting=False, private_inner_scoring=False, endpoint_scoring=False,
        raw_calibration_or_audit_response_access=False, response_magnitudes_recorded=False,
        original_attempt_modified=False, caps=vars(args))
    phase, last_event, current_partition = "scientific_imports", {}, {}
    attempts = {}
    training, validation, blocks = {}, {}, {}
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def stop(_signal, _frame):
        raise fitting.FitRunRefusal()

    with storage._exclusive_file(OUTPUT / "diagnostic_report.json") as stream:
        fitting._write(stream, report, args.report_byte_cap)

        def check(event=None):
            nonlocal last_event, current_partition
            if event is not None:
                last_event = fitting._event_description(event)
            budget(event)
            if (getattr(event, "operation", None) == "qualify_response_variance"
                    and getattr(event, "when", None) == "before"):
                key = event.group, event.stage
                index = attempts.get(key, 0)
                attempts[key] = index + 1
                if index < len(training):
                    current_partition = dict(kind="training", folds=tuple(training)[index])
                elif index < len(training) + len(validation):
                    current_partition = dict(kind="validation", label=tuple(validation)[index - len(training)])
                else:
                    offset = index - len(training) - len(validation)
                    require(0 <= offset < len(blocks))
                    current_partition = dict(kind="uncertainty_block", label=tuple(blocks)[offset])

        def stage(name):
            nonlocal phase
            phase = name
            report["phase"] = name
            fitting._write(stream, report, args.report_byte_cap)
            check()

        try:
            for sig in saved:
                signal.signal(sig, stop)
            signal.setitimer(signal.ITIMER_REAL, max(.001, args.deadline_epoch - time.time()))
            import complete_development_inputs as inputs
            import nested_prediction as nested
            stage("original_stopped_attempt_binding")
            original = inputs.existing._json(fitting.REPORT_ROOT / "fit001/fit_report.json",
                                              args.metadata_byte_cap, check)
            require(original.get("attempt") == "fit001" and original.get("status") == "stopped"
                and original.get("feature_attempt") == "full001"
                and original.get("error_code") == "complete_nested_fitting_failed"
                and original.get("last_operation") == "qualify_response_variance"
                and original.get("private_inner_scoring_attempted") is False
                and original.get("checkpoint_published") is False)
            stage("same_complete_development_input_loading")
            source = inputs.load_complete_development_inputs(feature_attempt="full001",
                feature_byte_cap=args.feature_byte_cap, metadata_byte_cap=args.metadata_byte_cap,
                response_byte_cap=args.response_byte_cap, resource_check=check)
            require(source.complete_development_response_release is True)
            report["input_loading_seconds"] = time.monotonic() - started
            report["loader_dimensions"] = dict(source.operational_dimensions)
            training, validation, blocks = nested._row_indices(source.design)
            # Same immutable response means as the full fitter. This numerical
            # preparation never mutates them; unused features can be released.
            design = source.design
            responses = {group: panel.stages for group, panel in source.responses.items()}
            del source
            stage("exact_response_preparation")
            nested._prepare_responses(design, responses, training, validation, blocks, check)
            report.update(status="completed_diagnosis", preparation_completed=True,
                          response_variance_refusal_reproduced=False)
        except BaseException as error:
            category = fitting._failure_category(error)
            reproduced = (phase == "exact_response_preparation"
                and category["numerical_failure_code"] == "response_partition_variance_invalid")
            report.update(category)
            report.update(status="completed_diagnosis" if reproduced else "stopped",
                error_code=phase + "_failed", preparation_completed=False,
                response_variance_refusal_reproduced=reproduced)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in saved.items():
                signal.signal(sig, handler)
            report.update(finished_utc=fitting.stamp(), elapsed_seconds=time.monotonic() - started,
                process_cpu_seconds=time.process_time() - cpu_started,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                completed_operation_counts=dict(budget.completed), last_operation=budget.last_operation,
                last_resource_event=last_event, response_partition=current_partition)
            fitting._write(stream, report, args.report_byte_cap)
    return 0 if report["status"] == "completed_diagnosis" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    for flag in ("rss-limit-kib", "metadata-byte-cap", "feature-byte-cap", "response-byte-cap", "report-byte-cap"):
        parser.add_argument("--" + flag, type=int, required=True)
    args = parser.parse_args(argv)
    try:
        fitting.require(bool(os.environ.get("SLURM_JOB_ID")) and "-ln" not in socket.gethostname()
            and os.environ.get("SLURM_CPUS_PER_TASK", "").isdecimal()
            and int(os.environ["SLURM_CPUS_PER_TASK"]) >= 1)
        fitting.require(all(os.environ.get(name) == "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")))
        fitting.require(math.isfinite(args.deadline_epoch) and 0 < args.deadline_epoch - time.time() <= 900)
        fitting.require(all(type(getattr(args, name)) is int and getattr(args, name) > 0 for name in
            ("rss_limit_kib", "metadata_byte_cap", "feature_byte_cap", "response_byte_cap", "report_byte_cap")))
        fitting.require(args.rss_limit_kib <= 6 * 1024 * 1024)
        code = _run(args)
    except BaseException:
        code = 1
    print(json.dumps({"status": "completed_diagnosis" if code == 0 else "stopped", "attempt": ATTEMPT}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
