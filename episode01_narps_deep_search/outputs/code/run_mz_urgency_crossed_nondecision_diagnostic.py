#!/usr/bin/env python3
"""Five coarse calls crossing only nondecision time between two saved states.

A/B batched vectors and scalarA remain preserved inputs. New calls are scalarB,
scalarC/batchedC and scalarD/batchedD. No optimization or mesh change occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
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
import run_mz_urgency_optimizer_trace as trace_producer
import analyze_mz_urgency_optimizer_trace as trace_analysis


optimization = endpoint.optimization
SOURCE, IMPLEMENTATION, PROBE = endpoint.SOURCE, endpoint.IMPLEMENTATION, endpoint.PROBE
TRACE = trace_analysis.TRACE
A_REFERENCE = SOURCE.parent / "20261005-MZ-urgency-endpoint-46615238-attempt002" / endpoint.REPORT_NAME
PHASE = "MZ0_urgency_saved_states_crossed_nondecision_five_call_diagnostic"
REPORT_NAME = "mz_urgency_crossed_nondecision_diagnostic.json"
MAX_SECONDS = 180
CALLS = (("B", "scalar"), ("C", "scalar"), ("C", "batched"), ("D", "scalar"), ("D", "batched"))


def crossed_points(a, b, names):
    if names != trace_analysis.NAMES or len(a) != 4 or len(b) != 4 or not np.isfinite([a, b]).all():
        raise ValueError("Cross only the four finite original M0 urgency parameters")
    index = names.index("nondecision_logit")
    if a[index] == b[index]:
        raise ValueError("The two selected states must have different nondecision coordinates")
    c, d = list(a), list(b)
    c[index], d[index] = b[index], a[index]
    return c, d


def operands(values, theta, mean, sd):
    nll = -np.asarray(values).sum()
    prior = .5*np.sum(((np.asarray(theta)-mean)/sd)**2)
    return {"nll": float(nll), "prior_penalty": float(prior), "conditional_objective": float(nll+prior)}


def saved_call(values, theta, mean, sd, origin, source_index):
    if len(values) != 128 or not np.isfinite(values).all():
        raise ValueError("A saved anchor must contain exactly128 finite trial scores")
    return {"origin": origin, "source_evaluation_index": source_index,
        "new_call": False, "returned": True, "finite128_vector": True,
        "trial_log_scores": deepcopy(values), **operands(values, theta, mean, sd)}


def compact_saved_geometry(geometry):
    result = deepcopy(geometry)
    rows = result.pop("rows")
    for key in ("response_present", "raw_bin_index", "density_bin_index", "safe_lookup_bin_index", "active"):
        result[key] = [row[key] for row in rows]
    return result


def validate_anchors(trace, reference, spec, source, probe, selected, rows, benchmark_report, check=lambda: None):
    evaluations, slots = trace_analysis.validate_trace(trace, check)
    names, mean, sd = endpoint.validate_probe(probe, spec, source, selected, rows, benchmark_report)
    identity, _ = optimization.probe_identity(2, selected)
    common = {**identity, "source_dir": str(SOURCE), "source_job_id": source["job_id"],
        "source_finished_utc": source["finished_utc"], "implementation": spec,
        "parameter_names": names, "conditional_prior_mean": mean.tolist(), "conditional_prior_sd": sd.tolist(),
        "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
        "probe_report": str(PROBE), "benchmark_report": str(optimization.BENCHMARK),
        "scalar_reference_report": str(optimization.benchmark.REFERENCE)}
    if (any(trace.get(key) != value for key, value in common.items())
            or trace.get("historical_start") != probe["attempts"][2] or trace["trial_slots"] != slots):
        raise ValueError("The exact saved trace must retain this source, conditional prior, participant and128 slots")
    expected_reference = {**common, "phase": endpoint.PHASE, "job_id": "46615238",
        "status": "urgency_endpoint_diagnostic_completed", "diagnostic_complete": True,
        "source_baseline_validated": True, "saved_primary_scores_replayed": True,
        "benchmark_validated": True, "terminal_probe_validated": True,
        "attempted_likelihood_calls": 6, "completed_likelihood_calls": 6,
        "scientific_optimizer_called": False, "refined_mesh_called": False}
    if any(reference.get(key) != value for key, value in expected_reference.items()):
        raise ValueError("Reuse only the matching completed scalarA endpoint reference")
    began, ended = (datetime.fromisoformat(reference[key]) for key in ("started_utc", "finished_utc"))
    if began.utcoffset() is None or ended.utcoffset() is None or ended < began:
        raise ValueError("The saved endpoint reference must be terminal")
    if len(reference.get("points", [])) != 3:
        raise ValueError("Preserve the completed original three-point reference")
    a, b, anchor = evaluations[270], evaluations[485], reference["points"][0]
    if (anchor.get("name") != "endpoint" or anchor.get("parameters") != a["parameters"]
            or anchor.get("coordinate_delta") != 0 or compact_saved_geometry(anchor["geometry"]) != a["geometry"]):
        raise ValueError("The saved scalarA geometry and parameters must exactly match trace270")
    for solver in ("scalar", "batched"):
        call = anchor["calls"][solver]
        if (call.get("returned") is not True or call.get("finite128_vector") is not True
                or call.get("trial_log_scores") != a["trial_log_scores"]
                or call.get("conditional_objective") != a["conditional_objective"]):
            raise ValueError("Both saved A vectors and objectives must retain exact original scalar/batched agreement")
    points = {}
    for name, index, record in (("A", 270, a), ("B", 485, b)):
        call = saved_call(record["trial_log_scores"], record["parameters"], mean, sd, "saved_trace", index)
        if any(call[key] != record[key] for key in ("nll", "prior_penalty", "conditional_objective")):
            raise ValueError("Saved anchor operands must retain the exact original initial conditional prior")
        points[name] = {"parameters": deepcopy(record["parameters"]), "geometry": deepcopy(record["geometry"]),
            "source_trace_evaluation_index": index, "calls": {"batched": call}}
    points["A"]["calls"]["scalar"] = saved_call(anchor["calls"]["scalar"]["trial_log_scores"], a["parameters"],
        mean, sd, "saved_endpoint_reference", None)
    c, d = crossed_points(a["parameters"], b["parameters"], names)
    for name, theta in (("C", c), ("D", d)):
        points[name] = {"parameters": theta, "geometry": None, "source_trace_evaluation_index": None, "calls": {}}
    return points, names, mean, sd


def summarize_crossed(points, mean, sd, slots):
    def flattened(name, solver):
        point, call = points[name], points[name]["calls"][solver]
        expected = operands(call["trial_log_scores"], point["parameters"], mean, sd)
        if any(call[key] != value for key, value in expected.items()):
            raise ValueError("Keep the same unchanged conditional objective arithmetic for every point")
        return {"evaluation_index": name, "parameters": point["parameters"], "geometry": point["geometry"],
            "trial_log_scores": call["trial_log_scores"], **expected}

    result = {"scalar_batched_comparisons": {}, "within_background_t0_comparisons": {}, "interaction": {},
        "interaction_sign": "(B-D)-(C-A): the nondecision-time effect under backgroundB minus that under backgroundA.",
        "interpretation": "Crossed conditional comparisons hold other parameters fixed within each background. Changing nondecision time also changes decision horizon and actual time step; these results do not isolate the bin lookup operator alone or establish the historical cause of optimizer failure. No optimizer, mesh, convergence or inference qualification."}
    for name in ("B", "C", "D"):
        scalar, batched = (points[name]["calls"][solver] for solver in ("scalar", "batched"))
        differences = np.asarray(batched["trial_log_scores"])-np.asarray(scalar["trial_log_scores"])
        result["scalar_batched_comparisons"][name] = {
            "batched_minus_scalar_trial_log_scores": differences.tolist(),
            "maximum_absolute_trial_log_score_difference": float(np.max(np.abs(differences))),
            "mean_trial_log_score_difference": float(differences.mean()),
            "conditional_objective_difference": batched["conditional_objective"]-scalar["conditional_objective"]}
    for solver in ("scalar", "batched"):
        records = {name: flattened(name, solver) for name in ("A", "B", "C", "D")}
        comparisons = []
        for left, right in (("A", "C"), ("D", "B")):
            comparison = trace_analysis.compare_evaluations(records[left], records[right], slots)
            comparison["left_point"] = comparison.pop("left_index")
            comparison["right_point"] = comparison.pop("right_index")
            comparisons.append(comparison)
        result["within_background_t0_comparisons"][solver] = comparisons
        trial_interaction = [(records["B"]["trial_log_scores"][i]-records["D"]["trial_log_scores"][i])
            -(records["C"]["trial_log_scores"][i]-records["A"]["trial_log_scores"][i]) for i in range(128)]
        result["interaction"][solver] = {
            key: (records["B"][key]-records["D"][key])-(records["C"][key]-records["A"][key])
            for key in ("nll", "prior_penalty", "conditional_objective")}
        result["interaction"][solver].update(trial_log_score_interaction=trial_interaction,
            negative_sum_trial_log_score_interaction=-math.fsum(trial_interaction))
    return result


def write_report(output, report):
    temporary = output / (REPORT_NAME+".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Crossed nondecision diagnostic interrupted by signal {signum}")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the unchanged original MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Bound the entire crossed diagnostic to180 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(key) != "1" for key in trace_analysis.THREAD_ENV):
        raise ValueError("Use one actually allocated CPU and six numerical thread limits of one")
    output = args.output_dir.resolve()
    protected = (SOURCE, PROBE.parent.parent, TRACE.parent, A_REFERENCE.parent,
                 optimization.BENCHMARK.parent, optimization.benchmark.REFERENCE.parent)
    if output.exists() or any(output == root.resolve() or root.resolve() in output.parents for root in protected):
        raise FileExistsError("Use a fresh output outside every preserved source/reference attempt")
    output.mkdir(parents=True)
    began, deadline = time.perf_counter(), time.monotonic()+args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_dir": str(SOURCE), "trace_report": str(TRACE), "scalarA_reference": str(A_REFERENCE),
        "probe_report": str(PROBE), "benchmark_report": str(optimization.BENCHMARK),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "diagnostic_complete": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "benchmark_validated": False, "saved_anchors_validated": False,
        "expected_new_likelihood_calls": 5, "attempted_new_likelihood_calls": 0, "completed_new_likelihood_calls": 0,
        "new_call_order": [], "points": {}, "summary": None,
        "A_reevaluated": False, "optimizer_called": False, "optimizer_settings_changed": False,
        "mesh_changed": False, "simulation_regenerated": False, "observed_choices_or_RTs_used": False,
        "population_hyperparameters_updated": False, "heldout_urgency_scores_computed": False,
        "full_fitter_solver_adopted": False, "automatic_retry": False, "full_recovery_pass": None,
        **{field: "not_evaluated" for field in optimization.scalar.CLAIM_FIELDS},
        "interpretation": "Exactly five new coarse calls at scalarB/scalarC/batchedC/scalarD/batchedD. A/B saved batched anchors and matching saved scalarA are unchanged. No optimizer, mesh or scientific inference change is selected.",
        "cost_accounting": "This process charges original-source replay, saved-anchor admission and five new calls once. Saved trace and endpoint reference costs remain historical and are not recharged."}

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter()-began
        write_report(output, report)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    complete = False
    try:
        checkpoint()
        check()
        spec = json.loads(args.implementation.read_text())
        if (spec["optimizer"]["maximum_sweeps"] != 500 or spec["urgency_solver"] != {
                "dt_seconds": .002, "spatial_intervals": 200,
                "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}):
            raise ValueError("Preserve original500 and the unchanged coarse .002/200 solver")
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
        scalar_reference = json.loads(optimization.benchmark.REFERENCE.read_text())
        benchmark_report = json.loads(optimization.BENCHMARK.read_text())
        optimization.validate_benchmark(benchmark_report, scalar_reference, spec, source, primary, selected, plan)
        report["benchmark_validated"] = True
        probe = json.loads(PROBE.read_text())
        trace = json.loads(TRACE.read_text())
        reference = json.loads(A_REFERENCE.read_text())
        identity, half = optimization.probe_identity(2, selected)
        rows = splits[half][0][identity["participant"]]["rows"]
        points, names, mean, sd = validate_anchors(trace, reference, spec, source, probe, selected, rows, benchmark_report, check)
        report.update(identity, saved_anchors_validated=True, points=points, parameter_names=names,
            conditional_prior_mean=mean.tolist(), conditional_prior_sd=sd.tolist(),
            source_trace_finished_utc=trace["finished_utc"], scalarA_reference_finished_utc=reference["finished_utc"],
            trial_slots=[[int(row["run"]), int(row["trial_order"])] for row in rows])
        del trace, reference
        check()
        data = shared.make_data(rows)
        for name in ("C", "D"):
            points[name]["geometry"] = trace_producer.compact_geometry(np.asarray(points[name]["parameters"]), names, data, spec)
        checkpoint()
        for name, solver in CALLS:
            check()
            point = points[name]
            if solver in point["calls"]:
                raise RuntimeError("A preserved anchor must never be replaced by a new call")
            call = {"origin": "new_coarse_likelihood", "new_call": True, "returned": False,
                "finite128_vector": False, "trial_log_scores": None, "elapsed_seconds": 0.,
                "nll": None, "prior_penalty": None, "conditional_objective": None}
            point["calls"][solver] = call
            report["new_call_order"].append({"point": name, "solver": solver})
            report["attempted_new_likelihood_calls"] += 1
            checkpoint()
            started = time.perf_counter()
            try:
                check()
                theta = np.asarray(point["parameters"])
                values = np.asarray(shared.trial_likelihood(theta, names, data, spec, "urgency") if solver == "scalar"
                                    else optimization.batched.trial_likelihood(theta, names, data, spec))
                call["returned"] = True
                if values.ndim == 1:
                    call["trial_log_scores"] = [float(value) if np.isfinite(value) else None for value in values]
                if values.shape != (128,) or not np.isfinite(values).all():
                    raise ValueError("Every selected new call must retain exactly128 finite trial scores")
                record = operands(values, theta, mean, sd)
                if not all(math.isfinite(value) for value in record.values()):
                    raise ValueError("Conditional objective operands must be finite")
                call.update(record, finite128_vector=True)
                report["completed_new_likelihood_calls"] += 1
            except Exception as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                call["elapsed_seconds"] = time.perf_counter()-started
                checkpoint()
            check()
        if report["completed_new_likelihood_calls"] != 5:
            raise RuntimeError("Completion requires exactly the five selected new calls")
        report["summary"] = summarize_crossed(points, mean, sd, report["trial_slots"])
        json.dumps(report["summary"], allow_nan=False)
        check()
        complete = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            complete = False
            report["error"] = "RuntimeError: Internal whole-process deadline exceeded before final publication"
        report.update(status="crossed_nondecision_diagnostic_completed" if complete else "technical_crossed_nondecision_failure",
            diagnostic_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        if not complete:
            report["summary"] = None
        checkpoint()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_crossed_nondecision_failure", diagnostic_complete=False, summary=None,
                error="RuntimeError: Internal whole-process deadline exceeded during final publication",
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
    print(json.dumps({key: report[key] for key in ("phase", "status", "diagnostic_complete",
        "attempted_new_likelihood_calls", "completed_new_likelihood_calls", "elapsed_seconds", "error") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
