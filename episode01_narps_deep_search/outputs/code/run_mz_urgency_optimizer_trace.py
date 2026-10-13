#!/usr/bin/env python3
"""Trace the unchanged optimizer's single original probe2/start2 trajectory.

The likelihood adapter records the existing evaluations and returns the exact
original array. No extra likelihood call, endpoint reevaluation, optimizer
setting, fitted population, numerical mesh or inference is selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np

from qualify_assets import require_compute
import run_mz_urgency_endpoint_diagnostic as endpoint


optimization = endpoint.optimization
SOURCE, IMPLEMENTATION, PROBE = endpoint.SOURCE, endpoint.IMPLEMENTATION, endpoint.PROBE
ORIGINAL_START_INDEX = 2
PHASE = "MZ0_urgency_probe2_original_start2_unchanged_optimizer_trace"
REPORT_NAME = "mz_urgency_optimizer_trace.json"
MAX_SECONDS = 1200


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def compact_geometry(theta, names, data, spec):
    """Preserve the previously reviewed grid arithmetic in compact arrays."""
    geometry = endpoint.grid_metadata(theta, names, data, spec)
    rows = geometry.pop("rows")
    for key in ("response_present", "raw_bin_index", "density_bin_index", "safe_lookup_bin_index", "active"):
        geometry[key] = [row[key] for row in rows]
    return geometry


def make_trace_likelihood(likelihood, mean, sd, evaluations):
    """Observe existing calls without changing returned values or exceptions.

    Computing the original scalar sum/prior from the returned values is only
    trace arithmetic. The unchanged conditional helper still performs its own
    original arithmetic. There is no additional likelihood/objective call.
    """
    def traced(theta, names, data, spec):
        record = {"evaluation_index": len(evaluations),
            "parameters": [number(value) for value in theta],
            "parameters_finite": bool(np.isfinite(theta).all()),
            "likelihood_entered": True, "likelihood_returned": False,
            "likelihood_seconds": 0., "adapter_seconds": 0.,
            "trial_log_scores": None, "trial_log_score_shape": None,
            "trial_log_scores_finite": None, "nll": None, "nll_finite": None,
            "prior_penalty": None, "prior_penalty_finite": None,
            "conditional_objective": None, "conditional_objective_finite": None,
            "objective_arithmetic_recorded": False, "geometry": None}
        evaluations.append(record)
        began = time.perf_counter()
        try:
            try:
                values = likelihood(theta, names, data, spec)
                record["likelihood_returned"] = True
            except Exception as error:
                record["likelihood_error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                record["likelihood_seconds"] = time.perf_counter()-began
            # Never impose a new finite-vector rule or clip/replace values.
            # The original helper alone determines numerical-error behavior.
            array = np.asarray(values)
            record.update(trial_log_score_shape=list(array.shape),
                trial_log_scores=[number(value) for value in array.ravel()],
                trial_log_scores_finite=bool(np.isfinite(array).all()))
            nll = -values.sum()
            prior = .5*np.sum(((theta-mean)/sd)**2)
            objective = float(nll + prior)
            record.update(nll=number(nll), nll_finite=bool(np.isfinite(nll)),
                prior_penalty=number(prior), prior_penalty_finite=bool(np.isfinite(prior)),
                conditional_objective=number(objective), conditional_objective_finite=math.isfinite(objective),
                objective_arithmetic_recorded=True)
            try:
                record["geometry"] = compact_geometry(theta, names, data, spec)
            except Exception as error:
                # A trace failure must not be misclassified as a scientific
                # likelihood ValueError/FloatingPointError by the old helper.
                record["trace_metadata_error"] = f"{type(error).__name__}: {error}"
                raise RuntimeError("Compact trace geometry could not be recorded") from error
            return values
        finally:
            record["adapter_seconds"] = time.perf_counter()-began
    return traced


def summarize_trace(attempt, evaluations, historical_attempt):
    """Exact descriptive matches among recorded calls; no new tolerance."""
    finite_x = attempt.get("parameters_finite") is True
    finite_fun = attempt.get("objective_finite") is True
    at_x = [row for row in evaluations if finite_x and row["parameters_finite"]
            and row["parameters"] == attempt["parameters"]]
    at_fun = [row for row in evaluations if finite_fun and row["conditional_objective_finite"] is True
              and row["conditional_objective"] == attempt["objective"]]
    both = [row["evaluation_index"] for row in at_x if row in at_fun]
    compared = ("start_parameters", "parameters", "parameters_finite", "objective", "objective_finite",
                "success", "status", "message", "nfev", "njev", "nit", "likelihood_calls_attempted",
                "likelihood_calls_returned", "numerical_error_calls")
    return {"match_rule": "Exact recorded numeric equality; no tolerance and no reevaluation.",
        "returned_parameters_matching_evaluation_indices": [row["evaluation_index"] for row in at_x],
        "reported_objective_matching_evaluation_indices": [row["evaluation_index"] for row in at_fun],
        "returned_parameters_and_reported_objective_matching_evaluation_indices": both,
        "recorded_objectives_at_returned_parameters": [
            {"evaluation_index": row["evaluation_index"], "conditional_objective": row["conditional_objective"],
             "conditional_objective_finite": row["conditional_objective_finite"],
             "objective_minus_reported_fun": number(row["conditional_objective"]-attempt["objective"])
                if row["conditional_objective_finite"] is True and finite_fun else None}
            for row in at_x],
        "historical_comparison": {key: {"historical": historical_attempt.get(key), "traced": attempt.get(key),
            "exactly_equal": historical_attempt.get(key) == attempt.get(key)} for key in compared},
        "historical_parameter_differences": [number(new-old) for new, old in
            zip(attempt["parameters"], historical_attempt["parameters"])] if finite_x else None,
        "historical_objective_difference": number(attempt["objective"]-historical_attempt["objective"])
            if finite_fun else None,
        "interpretation": "Matches describe only this recorded single-start trajectory. Historical reproduction is not required or a qualification threshold. A returned unsuccessful x/fun pair remains unsuccessful even if values reproduce."}


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Optimizer trace interrupted by signal {signum}; completed evaluations remain retained")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the original fixed MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole single-start trace is bounded to1200 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(key) != "1" for key in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("One allocated CPU and one numerical thread per library are required")
    output = args.output_dir.resolve()
    protected = (SOURCE, PROBE.parent.parent, optimization.BENCHMARK.parent, optimization.benchmark.REFERENCE.parent)
    if output.exists() or any(output == path.resolve() or path.resolve() in output.parents for path in protected):
        raise FileExistsError("Use a fresh output outside all preserved source/probe/reference attempts")
    output.mkdir(parents=True)
    began, deadline = time.perf_counter(), time.monotonic()+args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": optimization.scalar.shared.SEED,
        "rival": "urgency", "source_dir": str(SOURCE), "probe_report": str(PROBE),
        "probe_job_id": endpoint.PROBE_JOB, "benchmark_report": str(optimization.BENCHMARK),
        "scalar_reference_report": str(optimization.benchmark.REFERENCE),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "trace_complete": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "benchmark_validated": False, "terminal_probe_validated": False,
        "original_start_index": ORIGINAL_START_INDEX, "local_start_index": 0,
        "expected_starts": 1, "completed_starts": 0, "optimizer_called": False,
        "attempts": [], "best_start": None, "evaluations": [], "comparison": None,
        "likelihood_calls_attempted": 0, "likelihood_calls_returned": 0, "likelihood_seconds": 0.,
        "recorded_evaluations": 0, "recorded_likelihood_returns": 0,
        "simulation_regenerated": False, "observed_choices_or_RTs_used": False,
        "population_hyperparameters_updated": False, "full_cohort_model_fitted": False,
        "heldout_urgency_scores_computed": False, "refined_mesh_called": False,
        "original_48_call_benchmark_repeated": False, "optimizer_settings_changed": False,
        "added_likelihood_evaluations": 0, "endpoint_reevaluated_after_optimizer": False,
        "full_fitter_solver_adopted": False, "automatic_retry": False, "full_recovery_pass": None,
        "batched_solver_adoption_scope": "this_original_single_start_trace_only",
        **{field: "not_evaluated" for field in optimization.scalar.CLAIM_FIELDS},
        "interpretation": "Trace only original probe2/start2 from its original initialization, unchanged initial conditional prior and unchanged optimizer. Completion means a returned single-start trace, not optimizer success or a fitted population. No repair, fitted-support precision qualification or scientific inference is selected.",
        "cost_accounting": "This process includes full-source replay, admission and the one traced start. Original probe and benchmark costs remain historical; the coordinator charges the allocation once."}

    def checkpoint():
        report["completed_starts"] = sum(row["returned"] for row in report["attempts"])
        report["optimizer_called"] = any(row["optimizer_entered"] for row in report["attempts"])
        for key in ("likelihood_calls_attempted", "likelihood_calls_returned", "likelihood_seconds"):
            report[key] = sum(row[key] for row in report["attempts"])
        report["recorded_evaluations"] = len(report["evaluations"])
        report["recorded_likelihood_returns"] = sum(row["likelihood_returned"] for row in report["evaluations"])
        report["elapsed_seconds"] = time.perf_counter()-began
        write_report(output, report)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    complete = False
    try:
        checkpoint()
        check()
        spec = json.loads(args.implementation.read_text())
        if (spec["optimizer"]["maximum_sweeps"] != 500 or spec["parameters"]["urgency"]["bounds"] != [0., 2.]
                or spec["urgency_solver"] != {"dt_seconds": .002, "spatial_intervals": 200,
                    "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}):
            raise ValueError("Keep the original implementation, supports, optimizer and coarse mesh")
        report["implementation"] = spec
        shared = optimization.scalar.shared
        source = json.loads((SOURCE / optimization.scalar.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
            source_finished_utc=source["finished_utc"], source_participants=108, source_runs=432, source_trials=27648)
        splits = [shared.load_split(events, list(train), list(held)) for train, held in optimization.scalar.HALVES]
        primary, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        selected, plan = optimization.benchmark.probe_plan(people, splits, spec)
        reference = json.loads(optimization.benchmark.REFERENCE.read_text())
        benchmark_report = json.loads(optimization.BENCHMARK.read_text())
        optimization.validate_benchmark(benchmark_report, reference, spec, source, primary, selected, plan)
        report["benchmark_validated"] = True
        probe = json.loads(PROBE.read_text())
        identity, half = optimization.probe_identity(2, selected)
        rows = splits[half][0][identity["participant"]]["rows"]
        names, mean, sd = endpoint.validate_probe(probe, spec, source, selected, rows, benchmark_report)
        _, original_starts = optimization.scalar.initialization_vectors("M0", spec)
        report.update(identity, terminal_probe_validated=True, terminal_probe_finished_utc=probe["finished_utc"],
            historical_start=probe["attempts"][ORIGINAL_START_INDEX],
            historical_probe_elapsed_seconds=probe["elapsed_seconds"],
            parameter_names=names, conditional_prior_mean=mean.tolist(), conditional_prior_sd=sd.tolist(),
            original_start_parameters=original_starts[ORIGINAL_START_INDEX].tolist(),
            trial_slots=[[int(row["run"]), int(row["trial_order"])] for row in rows])
        check()
        data = shared.make_data(rows)
        checkpoint()
        adapter = make_trace_likelihood(optimization.batched.trial_likelihood, mean, sd, report["evaluations"])
        report["best_start"] = optimization.optimize_subject_block(data, names, spec, mean, sd,
            [original_starts[ORIGINAL_START_INDEX]], likelihood=adapter, attempts=report["attempts"],
            check=check, checkpoint=checkpoint)
        check()
        if len(report["attempts"]) != 1 or report["attempts"][0]["returned"] is not True:
            raise RuntimeError("The one selected original start must return before trace completion")
        report["comparison"] = summarize_trace(report["attempts"][0], report["evaluations"], report["historical_start"])
        json.dumps(report["comparison"], allow_nan=False)
        check()
        complete = True
    except Exception as error:
        report["incomplete_reason"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            complete = False
            report["incomplete_reason"] = "RuntimeError: Internal whole-process deadline exceeded before final publication"
        report.update(status="urgency_optimizer_trace_completed" if complete else "urgency_optimizer_trace_incomplete",
            trace_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        if complete and time.monotonic() >= deadline:
            report.update(status="urgency_optimizer_trace_incomplete", trace_complete=False,
                incomplete_reason="RuntimeError: Internal whole-process deadline exceeded during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "trace_complete", "completed_starts",
        "likelihood_calls_attempted", "recorded_evaluations", "elapsed_seconds", "incomplete_reason") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["trace_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
