#!/usr/bin/env python3
"""Saved-JSON-only arithmetic for the fixed completed490-call urgency trace.

No scientific likelihood, solver, optimizer, source event or model module is
imported. Recorded call order is not interpreted as accepted optimizer steps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import time


EPISODE = Path(__file__).resolve().parents[2]
TRACE = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-urgency-optimizer-trace-46617879/mz_urgency_optimizer_trace.json"
TRACE_PHASE = "MZ0_urgency_probe2_original_start2_unchanged_optimizer_trace"
PHASE = "MZ0_urgency_original_start2_saved_trace_numerical_accounting"
NAMES = ["kappa", "log_a", "nondecision_logit", "urgency"]
PAIRS = ((270, 285), (270, 485))
DETAIL_PAIRS = ((270, 273), (285, 288), (485, 488))
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
BIN_KEYS = ("raw_bin_index", "density_bin_index", "safe_lookup_bin_index")
GEOMETRY_KEYS = ("t0_seconds", "decision_horizon_seconds", "n_steps", "actual_dt_seconds",
    "spatial_low", "spatial_lower_weight", "spatial_upper_weight", "z", "actual_unique_drift_count")
MAX_SECONDS, MAX_INPUT_BYTES = 60, 64*1024*1024


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_trace(report, check=lambda: None):
    expected = {"phase": TRACE_PHASE, "job_id": "46617879", "status": "urgency_optimizer_trace_completed",
        "trace_complete": True, "original_start_index": 2, "local_start_index": 0,
        "completed_starts": 1, "parameter_names": NAMES, "model": "M0", "rival": "urgency",
        "probe_index": 2, "task_version": "ER", "training_runs": [1, 4], "excluded_scoring_runs": [2, 3],
        "likelihood_calls_attempted": 490, "likelihood_calls_returned": 490,
        "recorded_evaluations": 490, "recorded_likelihood_returns": 490,
        "added_likelihood_evaluations": 0, "endpoint_reevaluated_after_optimizer": False,
        "source_baseline_validated": True, "saved_primary_scores_replayed": True,
        "benchmark_validated": True, "terminal_probe_validated": True}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Only the selected completed46617879 trace is admitted")
    began, ended = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
    if began.utcoffset() is None or ended.utcoffset() is None or ended < began:
        raise ValueError("The fixed trace must have terminal UTC timestamps")
    attempts, evaluations, slots = report.get("attempts"), report.get("evaluations"), report.get("trial_slots")
    if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(evaluations, list) or len(evaluations) != 490:
        raise ValueError("Retain exactly one returned start and all490 recorded evaluations")
    attempt = attempts[0]
    if any(attempt.get(key) != value for key, value in {"returned": True, "start_index": 0,
            "success": False, "status": 2, "message": "ABNORMAL_TERMINATION_IN_LNSRCH",
            "nfev": 490, "njev": 98, "nit": 10, "numerical_error_calls": 0,
            "parameters_finite": True, "objective_finite": True}.items()):
        raise ValueError("Retain the observed unsuccessful original-start2 return")
    if (not isinstance(slots, list) or len(slots) != 128
            or {tuple(slot) for slot in slots} != {(run, trial) for run in (1, 4) for trial in range(1, 65)}):
        raise ValueError("Retain the exact128 selected training slots")
    for index, row in enumerate(evaluations):
        check()
        if (row.get("evaluation_index") != index or row.get("likelihood_returned") is not True
                or row.get("parameters_finite") is not True or row.get("trial_log_scores_finite") is not True
                or row.get("objective_arithmetic_recorded") is not True
                or row.get("trial_log_score_shape") != [128]
                or len(row.get("parameters", [])) != 4 or not all(map(finite, row["parameters"]))
                or len(row.get("trial_log_scores", [])) != 128 or not all(map(finite, row["trial_log_scores"]))
                or any(row.get(key+"_finite") is not True or not finite(row.get(key))
                       for key in ("nll", "prior_penalty", "conditional_objective"))):
            raise ValueError("Each recorded call must preserve finite parameters,128 scores and original objective operands")
        geometry = row["geometry"]
        if (any(not finite(geometry.get(key)) for key in GEOMETRY_KEYS)
                or type(geometry["n_steps"]) is not int or geometry["n_steps"] <= 0
                or geometry["actual_dt_seconds"] <= 0
                or any(not isinstance(geometry.get(key), list) or len(geometry[key]) != 128
                       for key in (*BIN_KEYS, "active", "response_present"))):
            raise ValueError("Retain complete compact geometry for every selected trial")
        for trial in range(128):
            present = geometry["response_present"][trial]
            if (type(present) is not bool
                    or present != evaluations[0]["geometry"]["response_present"][trial]
                    or (present and (type(geometry["active"][trial]) is not bool
                        or any(type(geometry[key][trial]) is not int for key in BIN_KEYS)))
                    or (not present and any(geometry[key][trial] is not None for key in (*BIN_KEYS, "active")))):
                raise ValueError("Trial response identities and corresponding bin/active arrays must remain aligned")
    at_x = [row["evaluation_index"] for row in evaluations if row["parameters"] == attempt["parameters"]]
    at_fun = [row["evaluation_index"] for row in evaluations if row["conditional_objective"] == attempt["objective"]]
    joint = sorted(set(at_x) & set(at_fun))
    comparison = report["comparison"]
    if (at_x != [270, 285] or at_fun != [485] or joint
            or comparison["returned_parameters_matching_evaluation_indices"] != at_x
            or comparison["reported_objective_matching_evaluation_indices"] != at_fun
            or comparison["returned_parameters_and_reported_objective_matching_evaluation_indices"] != joint):
        raise ValueError("Derived returned-x/fun locations must exactly reproduce the existing trace comparison")
    for recorded, index in zip(comparison["recorded_objectives_at_returned_parameters"], at_x):
        objective = evaluations[index]["conditional_objective"]
        if (recorded["evaluation_index"] != index or recorded["conditional_objective"] != objective
                or recorded["objective_minus_reported_fun"] != objective-attempt["objective"]):
            raise ValueError("Preserve existing returned-x objective comparison values")
    if len(comparison["recorded_objectives_at_returned_parameters"]) != 2:
        raise ValueError("Retain both repeated returned-x evaluations")
    return evaluations, slots


def geometry_changes(left, right):
    a, b = left["geometry"], right["geometry"]
    active = [a["active"][i] != b["active"][i] for i in range(128)]
    bins = [any(a[key][i] != b[key][i] for key in BIN_KEYS) for i in range(128)]
    return active, bins


def compare_evaluations(left, right, trial_slots):
    active, bins = geometry_changes(left, right)
    differences = [b-a for a, b in zip(left["trial_log_scores"], right["trial_log_scores"])]
    groups = {key: [] for key in ("active_changed", "bin_changed_without_active_change", "unchanged_active_and_bins")}
    rows = []
    for index, difference in enumerate(differences):
        group = "active_changed" if active[index] else (
            "bin_changed_without_active_change" if bins[index] else "unchanged_active_and_bins")
        groups[group].append(index)
        rows.append({"row_index": index, "trial_slot": trial_slots[index],
            "right_minus_left_trial_log_score": difference, "group": group,
            "active_changed": active[index], "any_bin_changed": bins[index],
            "left_active": left["geometry"]["active"][index], "right_active": right["geometry"]["active"][index],
            **{side+"_"+key: row["geometry"][key][index] for side, row in (("left", left), ("right", right))
               for key in BIN_KEYS}})
    delta_nll = right["nll"]-left["nll"]
    delta_prior = right["prior_penalty"]-left["prior_penalty"]
    delta_objective = right["conditional_objective"]-left["conditional_objective"]
    return {"left_index": left["evaluation_index"], "right_index": right["evaluation_index"],
        "parameter_differences": [b-a for a, b in zip(left["parameters"], right["parameters"])],
        "delta_nll": delta_nll, "delta_prior_penalty": delta_prior, "delta_conditional_objective": delta_objective,
        "negative_sum_trial_log_score_differences": -math.fsum(differences),
        "nll_difference_minus_negative_score_difference_sum": delta_nll+math.fsum(differences),
        "objective_difference_minus_nll_and_prior_difference": delta_objective-(delta_nll+delta_prior),
        "exact_equal": {key: left[key] == right[key] for key in
            ("parameters", "trial_log_scores", "nll", "prior_penalty", "conditional_objective", "geometry")},
        "geometry": {key: {"left": left["geometry"][key], "right": right["geometry"][key],
            "difference": right["geometry"][key]-left["geometry"][key]} for key in GEOMETRY_KEYS},
        "active_changed_count": sum(active), "any_bin_changed_count": sum(bins),
        "group_precedence": "Active changes first; remaining bin changes second; all other rows third. All128 rows belong to exactly one group.",
        "groups": {key: {"row_indices": indices, "count": len(indices),
            "negative_sum_trial_log_score_differences": -math.fsum(differences[i] for i in indices)}
            for key, indices in groups.items()}, "rows": rows,
        "interpretation": "Signed right-minus-left differences and disjoint score accounting, not a causal decomposition. Parameter and prior changes remain separate; a bin-change association does not establish why a line search failed."}


def summarize_packets(evaluations, names, check=lambda: None):
    packets, issues = [], []
    for offset in range(0, 490, 5):
        check()
        base = evaluations[offset]
        coordinates = []
        for coordinate, row in enumerate(evaluations[offset+1:offset+5]):
            changed = [i for i, (a, b) in enumerate(zip(base["parameters"], row["parameters"])) if a != b]
            if changed != [coordinate]:
                issues.append({"base_index": offset, "evaluation_index": row["evaluation_index"],
                    "expected_coordinate": names[coordinate], "observed_changed_coordinates": [names[i] for i in changed]})
                continue
            step = row["parameters"][coordinate]-base["parameters"][coordinate]
            active, bins = geometry_changes(base, row)
            coordinates.append({"coordinate": names[coordinate], "evaluation_index": row["evaluation_index"],
                "actual_recorded_step": step,
                "recorded_forward_slope": (row["conditional_objective"]-base["conditional_objective"])/step,
                "n_steps_changed": row["geometry"]["n_steps"] != base["geometry"]["n_steps"],
                "n_steps_difference": row["geometry"]["n_steps"]-base["geometry"]["n_steps"],
                "active_changed_count": sum(active), "any_bin_changed_count": sum(bins),
                "bin_changed_without_active_change_count": sum(b and not a for a, b in zip(active, bins))})
        packets.append({"base_index": offset, "parameters": base["parameters"], "nll": base["nll"],
            "prior_penalty": base["prior_penalty"], "conditional_objective": base["conditional_objective"],
            "t0_seconds": base["geometry"]["t0_seconds"], "n_steps": base["geometry"]["n_steps"],
            "actual_dt_seconds": base["geometry"]["actual_dt_seconds"], "coordinates": coordinates})
    return {"pattern_resolved": not issues, "expected_packets": 98,
        "pattern": "Recorded contiguous base plus four single-coordinate moves, in the recorded parameter-name order. No epsilon is assumed.",
        "issues": issues, "packets": packets if not issues else [],
        "interpretation": "These are verified evaluation packets, not accepted iterations, line-search acceptance decisions or Wolfe-condition checks."}


def analyze(report, check=lambda: None):
    evaluations, slots = validate_trace(report, check)
    pairs = [compare_evaluations(evaluations[a], evaluations[b], slots) for a, b in PAIRS]
    packet_summary = summarize_packets(evaluations, report["parameter_names"], check)
    detailed = [compare_evaluations(evaluations[a], evaluations[b], slots) for a, b in DETAIL_PAIRS
                ] if packet_summary["pattern_resolved"] else []
    check()
    return {"source_finished_utc": report["finished_utc"], "source_job_id": report["job_id"],
        "source_attempt_return": report["attempts"][0], "existing_trace_comparison": report["comparison"],
        "fixed_pairs": pairs, "finite_difference_packets": packet_summary,
        "detailed_nondecision_pairs": detailed,
        "detailed_nondecision_pairs_withheld": not packet_summary["pattern_resolved"],
        "interpretation": "The returned-x/reported-fun pairing and local grid/bin associations are distinct observations. Determinism is assessed only at the recorded repeated points. Matching or differing records do not qualify optimizer convergence, global solver equivalence, fitted-support mesh precision, causal explanation or any scientific inference."}


def interrupted(signum, frame):
    raise RuntimeError(f"Saved-trace analysis interrupted by signal {signum}")


def run(args):
    hostname = socket.gethostname()
    if ("-ln" in hostname or not os.environ.get("SLURM_JOB_ID")
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or any(os.environ.get(key) != "1" for key in THREAD_ENV)):
        raise RuntimeError("Use one actually allocated compute CPU and all six numerical thread limits of one")
    if args.trace_report.resolve() != TRACE.resolve():
        raise ValueError("Read only the fixed completed46617879 trace")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole report-only analysis is bounded to60 seconds")
    output = args.output.resolve()
    temporary = output.with_name(output.name+".partial")
    if (output.exists() or temporary.exists() or output == TRACE.parent or TRACE.parent in output.parents
            or EPISODE / "outputs" not in output.parents):
        raise FileExistsError("Use a fresh episode-output JSON outside the preserved trace attempt")
    output.parent.mkdir(parents=True, exist_ok=True)
    began, deadline = time.perf_counter(), time.monotonic()+args.max_seconds
    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "trace_report": str(TRACE), "job_id": os.environ["SLURM_JOB_ID"], "hostname": hostname,
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "status": "running", "analysis_complete": False, "analysis": None,
        "new_likelihood_calls": 0, "new_optimizer_calls": 0, "source_events_read": False,
        "optimizer_settings_changed": False, "mesh_changed": False,
        "optimizer_convergence_qualified": False, "causal_claim": False,
        "cost_accounting": "Only this saved-report analysis is timed here; previous trace/probe/fit durations are historical and are not added as new compute charges."}

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process saved-trace analysis deadline exceeded")

    published = False

    def publish():
        nonlocal published
        report["elapsed_seconds"] = time.perf_counter()-began
        with temporary.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if published:
            temporary.replace(output)
        else:
            os.link(temporary, output)
            temporary.unlink()
            published = True

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    complete = False
    try:
        check()
        with args.trace_report.open("rb") as stream:
            contents = stream.read(MAX_INPUT_BYTES+1)
        if len(contents) > MAX_INPUT_BYTES:
            raise ValueError("The fixed trace exceeds the selected64MiB report-read envelope")
        report["input_bytes"] = len(contents)
        source = json.loads(contents)
        check()
        report["analysis"] = analyze(source, check)
        json.dumps(report["analysis"], allow_nan=False)
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
            report["error"] = "RuntimeError: Whole-process deadline exceeded before final publication"
        report.update(status="saved_trace_analysis_completed" if complete else "technical_saved_trace_analysis_failure",
            analysis_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        if not complete:
            report["analysis"] = None
        publish()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_saved_trace_analysis_failure", analysis_complete=False, analysis=None,
                error="RuntimeError: Whole-process deadline exceeded during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            publish()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "analysis_complete", "elapsed_seconds", "error")
        if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["analysis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
