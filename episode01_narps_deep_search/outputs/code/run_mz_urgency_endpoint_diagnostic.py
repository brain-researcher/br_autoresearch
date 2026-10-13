#!/usr/bin/env python3
"""Three-point local numerical diagnosis of one fixed unsuccessful endpoint.

The constant-objective optimizer call records the installed finite-difference
stencil. It never evaluates the scientific likelihood or refits an endpoint.
Exactly three scalar and three batched coarse likelihood calls are selected.
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
from scipy.optimize import minimize
from scipy.special import expit

from qualify_assets import require_compute
import run_mz_urgency_optimization_cost_pilot as optimization


SOURCE, IMPLEMENTATION = optimization.SOURCE, optimization.IMPLEMENTATION
PROBE_JOB = "46612782"
PROBE = SOURCE.parent / f"20261005-MZ-urgency-optimization-cost-{PROBE_JOB}" / "probe2" / optimization.REPORT_NAME
PROBE_FINISHED = "2026-10-05T12:29:52.778568+00:00"
ENDPOINT = [-0.0067977447447558695, 0.03763003832812087, -2.6117007993902077, 0.5889305026961807]
REPORTED_OBJECTIVE = 14.594092987082634
ATTEMPT_CALLS = (350, 450, 490)
PHASE = "MZ0_urgency_probe2_fixed_endpoint_local_numerical_diagnostic"
REPORT_NAME = "mz_urgency_endpoint_diagnostic.json"
MAX_SECONDS = 300


def record_stencil(theta, names, spec, check=lambda: None):
    """Observe SciPy's actual bounded finite-difference step; never guess eps."""
    theta = np.asarray(theta, dtype=float)
    coordinate = names.index("nondecision_logit")
    bounds = [spec["parameters"][name]["bounds"] for name in names]
    if theta.shape != (len(names),) or not np.isfinite(theta).all():
        raise ValueError("The recorded endpoint must be a finite parameter vector")
    if any(not low <= value <= high for value, (low, high) in zip(theta, bounds)):
        raise ValueError("The endpoint must remain within the original bounds")
    vectors = []

    def constant(candidate):
        check()
        vectors.append(np.asarray(candidate, dtype=float).copy())
        return 0.

    optimizer = spec["optimizer"]
    options = {"maxiter": optimizer["subject_maximum_iterations"],
               "ftol": optimizer["subject_ftol"], "gtol": optimizer["subject_gtol"]}
    result = minimize(constant, theta.copy(), method="L-BFGS-B", bounds=bounds, options=options)
    check()
    if not vectors or not np.array_equal(vectors[0], theta):
        raise ValueError("The recording call must start at the exact unchanged endpoint")
    candidates = []
    for vector in vectors[1:]:
        changed = np.flatnonzero(vector != theta)
        if changed.tolist() == [coordinate] and not any(np.array_equal(vector, old) for old in candidates):
            candidates.append(vector)
    if len(candidates) != 1:
        raise ValueError("The installed unchanged optimizer did not expose one identifiable coordinate perturbation")
    perturbed = candidates[0]
    step = float(perturbed[coordinate] - theta[coordinate])
    if not math.isfinite(step) or step == 0:
        raise ValueError("The observed effective perturbation must be finite and nonzero")
    opposite = theta.copy()
    opposite[coordinate] = theta[coordinate] - step
    low, high = bounds[coordinate]
    if (not low <= perturbed[coordinate] <= high or not low <= opposite[coordinate] <= high
            or float(opposite[coordinate] - theta[coordinate]) != -step):
        raise ValueError("The same effective opposite-side diagnostic step is not representable within the original bounds")
    return {"coordinate": "nondecision_logit", "coordinate_index": coordinate,
        "method": "L-BFGS-B", "bounds": bounds, "options": options,
        "recording_vectors": [vector.tolist() for vector in vectors], "effective_step": step,
        "recording_optimizer": {"success": bool(result.success), "status": int(result.status),
            "message": str(result.message), "nfev": int(result.nfev), "nit": int(result.nit)},
        "points": [{"name": name, "parameters": vector.tolist(),
                    "coordinate_delta": float(vector[coordinate]-theta[coordinate])}
                   for name, vector in (("endpoint", theta), ("optimizer_perturbation", perturbed),
                                        ("opposite_diagnostic", opposite))],
        "interpretation": "One installed constant-objective recording call, using unchanged options/bounds. Opposite side is a declared diagnostic, not an additional optimizer start or new numerical setting."}


def grid_metadata(theta, names, data, spec):
    """Mirror the scalar solver's indexing arithmetic without solving a PDE."""
    params = dict(zip(names, theta))
    t0 = 4*expit(params["nondecision_logit"])
    horizon = 4-t0
    dt, intervals = spec["urgency_solver"]["dt_seconds"], spec["urgency_solver"]["spatial_intervals"]
    n_steps = math.ceil(horizon/dt)
    edges = np.linspace(0., horizon, n_steps+1)
    z, h = expit(params.get("rho", 0.)), 1/intervals
    fractional = z/h
    low = int(np.floor(fractional))
    responded = np.asarray(data["choice"]) != -1
    rows = []
    for index in range(len(responded)):
        if not responded[index]:
            rows.append({"row_index": index, "response_present": False, "decision_time": None,
                "raw_bin_index": None, "density_bin_index": None, "safe_lookup_bin_index": None, "active": None})
            continue
        decision_time = float(data["rt"][index]-t0)
        raw = int(np.searchsorted(edges, decision_time, side="right")-1)
        slot = min(raw, n_steps-1)
        rows.append({"row_index": index, "response_present": True, "decision_time": decision_time,
            "raw_bin_index": raw, "density_bin_index": slot, "safe_lookup_bin_index": max(slot, 0),
            "active": bool(decision_time > 0 and decision_time <= edges[-1])})
    drift, _ = optimization.scalar.drift_and_horizon(np.asarray(theta), names, data)
    return {"t0_seconds": float(t0), "decision_horizon_seconds": float(horizon),
        "nominal_dt_seconds": dt, "n_steps": n_steps, "actual_dt_seconds": float(horizon/n_steps),
        "spatial_intervals": intervals, "z": float(z), "spatial_h": h,
        "spatial_fractional_position": float(fractional), "spatial_low": low,
        "spatial_lower_array_index": low-1, "spatial_lower_weight": float(1-(fractional-low)),
        "spatial_upper_array_index": low, "spatial_upper_weight": float(fractional-low),
        "spatial_upper_mass_added": bool(fractional > low),
        "actual_unique_drift_count": int(len(np.unique(drift))), "rows": rows}


def conditional_objective(values, theta, mean, sd):
    return float(-np.asarray(values).sum() + .5*np.sum(((np.asarray(theta)-mean)/sd)**2))


def validate_probe(probe, spec, source, selected, rows, benchmark_report):
    """Bind the exact completed cost probe, all three starts and chosen endpoint."""
    identity, half = optimization.probe_identity(2, selected)
    names, starts = optimization.scalar.initialization_vectors("M0", spec)
    priors = [spec["parameters"][name] for name in names]
    mean = np.array([prior["population_center"] for prior in priors])
    initial_sd = np.array([prior["individual_scale_center"] for prior in priors])
    sd = np.exp(np.log(initial_sd))
    expected = {**identity, "analysis_id": "behavioral_mechanism_kill_test", "phase": optimization.PHASE,
        "status": "urgency_conditional_cost_probe_completed", "generating_model": "MZ", "replicate_index": 0,
        "seed": optimization.scalar.shared.SEED, "rival": "urgency", "implementation": spec,
        "source_dir": str(SOURCE), "source_report": str(SOURCE / optimization.scalar.SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "source_participants": 108, "source_runs": 432, "source_trials": 27648,
        "benchmark_report": str(optimization.BENCHMARK),
        "scalar_reference_report": str(optimization.benchmark.REFERENCE),
        "benchmark_finished_utc": benchmark_report["finished_utc"],
        "finished_utc": PROBE_FINISHED, "internal_seconds": 3000,
        "parameter_names": names, "conditional_prior_mean": mean.tolist(),
        "conditional_prior_sd": sd.tolist(), "original_initial_sd": initial_sd.tolist(),
        "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
        "selected_probe_participants": selected, "expected_starts": 3, "completed_starts": 3,
        "likelihood_calls_attempted": 1290, "likelihood_calls_returned": 1290,
        "batched_solver_adoption_scope": "this_conditional_cost_pilot_only"}
    if (any(probe.get(key) != value for key, value in expected.items()) or str(probe.get("job_id")) != PROBE_JOB
            or len(rows) != 128 or half != 0 or identity["task_version"] != "ER"
            or any(key in probe for key in ("incomplete_reason", "active_call"))):
        raise ValueError("Only the selected completed probe2 on its exact source/spec/rows is authorized")
    for key in ("probe_complete", "conditional_optimization_complete", "source_baseline_validated",
                "saved_primary_scores_replayed", "benchmark_validated", "optimizer_called"):
        if probe.get(key) is not True:
            raise ValueError("The original cost probe must be fully completed and validated")
    for key in ("simulation_regenerated", "simulation_copied", "observed_choices_or_RTs_used",
                "population_hyperparameters_updated", "full_cohort_model_fitted", "heldout_urgency_scores_computed",
                "actual_fitted_support_precision_evaluated", "full_fitter_solver_adopted", "wave_selected",
                "automatic_retry", "episode_terminal"):
        if probe.get(key) is not False:
            raise ValueError("Preserve the original cost-only scope")
    if (probe.get("full_recovery_pass") is not None
            or any(probe.get(key) != "not_evaluated" for key in optimization.scalar.CLAIM_FIELDS)
            or probe.get("saved_score_replay_absolute_tolerance") != optimization.scalar.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not optimization.benchmark.finite(probe.get("saved_score_replay_max_absolute_discrepancy"))
            or not 0 <= probe["saved_score_replay_max_absolute_discrepancy"] <= optimization.scalar.shared.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("Require the preserved full-source replay and noninferential claims")
    began, ended = (datetime.fromisoformat(probe[key]) for key in ("started_utc", "finished_utc"))
    if began.utcoffset() is None or ended.utcoffset() is None or ended < began:
        raise ValueError("The selected probe must retain valid terminal timestamps")
    if not optimization.benchmark.finite(probe.get("elapsed_seconds")) or probe["elapsed_seconds"] <= 0:
        raise ValueError("Retain the terminal probe's positive measured elapsed cost")
    attempts = probe.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 3:
        raise ValueError("Retain the three complete original starts")
    for index, (attempt, start) in enumerate(zip(attempts, starts)):
        if (attempt.get("start_index") != index or attempt.get("start_parameters") != start.tolist()
                or attempt.get("optimizer_entered") is not True or attempt.get("returned") is not True
                or attempt.get("success") is not False or attempt.get("status") != 2
                or attempt.get("message") != "ABNORMAL_TERMINATION_IN_LNSRCH"
                or attempt.get("nfev") != ATTEMPT_CALLS[index]
                or attempt.get("likelihood_calls_attempted") != ATTEMPT_CALLS[index]
                or attempt.get("likelihood_calls_returned") != ATTEMPT_CALLS[index]
                or attempt.get("numerical_error_calls") != 0
                or attempt.get("objective_finite") is not True or attempt.get("parameters_finite") is not True
                or not optimization.benchmark.finite(attempt.get("objective"))
                or not isinstance(attempt.get("parameters"), list) or len(attempt["parameters"]) != len(names)
                or not all(optimization.benchmark.finite(value) for value in attempt["parameters"])
                or "interruption" in attempt):
            raise ValueError("All three original line-search failures and starts must remain unchanged")
    if (sum(attempt["likelihood_calls_attempted"] for attempt in attempts) != 1290
            or sum(attempt["likelihood_calls_returned"] for attempt in attempts) != 1290
            or min(range(3), key=lambda index: attempts[index]["objective"]) != 2
            or attempts[2]["parameters"] != ENDPOINT or attempts[2]["objective"] != REPORTED_OBJECTIVE):
        raise ValueError("The original first-minimum selection or fixed returned endpoint changed")
    expected_best = {"start_index": 2, "parameters": ENDPOINT, "parameters_finite": True,
        "objective": REPORTED_OBJECTIVE, "objective_finite": True, "success": False,
        "message": "ABNORMAL_TERMINATION_IN_LNSRCH"}
    if probe.get("best_start") != expected_best:
        raise ValueError("Retain the exact unsuccessful best_start, including its reported objective")
    return names, mean, sd


def summarize_points(points, reported_objective):
    """Diagnostic differences/slopes only: no acceptance threshold or gate."""
    endpoint = points[0]
    positive = next(point for point in points[1:] if point["coordinate_delta"] > 0)
    negative = next(point for point in points[1:] if point["coordinate_delta"] < 0)
    result = {"reported_optimizer_objective": reported_objective, "solvers": {}}
    for solver in ("scalar", "batched"):
        center = endpoint["calls"][solver]["conditional_objective"]
        result["solvers"][solver] = {
            "recomputed_endpoint_objective": center,
            "endpoint_objective_minus_optimizer_report": center-reported_objective,
            "forward_slope": (positive["calls"][solver]["conditional_objective"]-center)/positive["coordinate_delta"],
            "backward_slope": (center-negative["calls"][solver]["conditional_objective"])/(-negative["coordinate_delta"]),
            "local_comparisons": [{"point": point["name"], "coordinate_delta": point["coordinate_delta"],
                "point_minus_endpoint_trial_log_scores": (
                    np.asarray(point["calls"][solver]["trial_log_scores"])
                    - np.asarray(endpoint["calls"][solver]["trial_log_scores"])).tolist()}
                for point in points[1:]]}
    result["scalar_batched_comparisons"] = []
    for point in points:
        difference = np.asarray(point["calls"]["batched"]["trial_log_scores"]) - np.asarray(point["calls"]["scalar"]["trial_log_scores"])
        result["scalar_batched_comparisons"].append({"point": point["name"],
            "batched_minus_scalar_trial_log_scores": difference.tolist(),
            "maximum_absolute_trial_log_score_difference": float(np.max(np.abs(difference))),
            "mean_trial_log_score_difference": float(difference.mean()),
            "conditional_objective_difference": point["calls"]["batched"]["conditional_objective"]-point["calls"]["scalar"]["conditional_objective"]})
    return result


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Endpoint diagnostic interrupted by signal {signum}; incomplete numerical evidence is retained")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the fixed original MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole endpoint diagnostic is bounded to300 seconds")
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
        "source_dir": str(SOURCE), "probe_report": str(PROBE), "probe_job_id": PROBE_JOB,
        "benchmark_report": str(optimization.BENCHMARK), "scalar_reference_report": str(optimization.benchmark.REFERENCE),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "diagnostic_complete": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "benchmark_validated": False, "terminal_probe_validated": False,
        "expected_likelihood_calls": 6, "attempted_likelihood_calls": 0, "completed_likelihood_calls": 0,
        "points": [], "summary": None, "scientific_optimizer_called": False,
        "constant_objective_recording_called": False, "simulation_regenerated": False,
        "observed_choices_or_RTs_used": False, "population_hyperparameters_updated": False,
        "heldout_urgency_scores_computed": False, "refined_mesh_called": False,
        "original_48_call_benchmark_repeated": False, "optimizer_settings_changed": False,
        "full_fitter_solver_adopted": False, "automatic_retry": False, "full_recovery_pass": None,
        **{field: "not_evaluated" for field in optimization.scalar.CLAIM_FIELDS},
        "interpretation": "Fixed unsuccessful probe2 endpoint and its observed finite-difference coordinate step, plus one declared opposite-side diagnostic. Scalar/batched differences and local slopes are numerical evidence only. The unsuccessful returned x and reported fun may disagree; retain that discrepancy without rejecting it. No full fit, new optimizer setting, precision threshold or inference is selected.",
        "cost_accounting": "elapsed_seconds charges this source replay, validation, cheap constant-objective recording and six selected likelihood calls once; the original cost probe and benchmark remain historical references."}

    def checkpoint():
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
        if spec["optimizer"]["maximum_sweeps"] != 500 or spec["urgency_solver"] != {
                "dt_seconds": .002, "spatial_intervals": 200,
                "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}:
            raise ValueError("Preserve original500 and the original coarse mesh")
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
        identity, half = optimization.probe_identity(2, selected)
        rows = splits[half][0][identity["participant"]]["rows"]
        names, mean, sd = validate_probe(probe, spec, source, selected, rows, benchmark_report)
        report.update(identity, terminal_probe_validated=True, terminal_probe_finished_utc=probe["finished_utc"],
            historical_probe_elapsed_seconds=probe["elapsed_seconds"],
            parameter_names=names, conditional_prior_mean=mean.tolist(), conditional_prior_sd=sd.tolist(),
            selected_endpoint=ENDPOINT, optimizer_reported_endpoint_objective=REPORTED_OBJECTIVE,
            trial_slots=[[int(row["run"]), int(row["trial_order"])] for row in rows])
        check()
        data = shared.make_data(rows)
        report["constant_objective_recording_called"] = True
        stencil = record_stencil(ENDPOINT, names, spec, check)
        report["stencil"] = stencil
        checkpoint()
        for definition in stencil["points"]:
            check()
            theta = np.asarray(definition["parameters"])
            point = {**definition, "geometry": grid_metadata(theta, names, data, spec), "calls": {}}
            report["points"].append(point)
            for solver in ("scalar", "batched"):
                check()
                call = {"attempted": True, "returned": False, "finite128_vector": False,
                        "elapsed_seconds": 0., "trial_log_scores": None, "conditional_objective": None}
                point["calls"][solver] = call
                report["attempted_likelihood_calls"] += 1
                checkpoint()
                started = time.perf_counter()
                try:
                    values = np.asarray(shared.trial_likelihood(theta, names, data, spec, "urgency") if solver == "scalar"
                                        else optimization.batched.trial_likelihood(theta, names, data, spec))
                    call["returned"] = True
                    if values.ndim == 1:
                        call["trial_log_scores"] = [float(value) if np.isfinite(value) else None for value in values]
                    if values.shape != (128,) or not np.isfinite(values).all():
                        raise ValueError("A diagnostic call must return exactly128 finite trial scores; no rows may be dropped")
                    call["finite128_vector"] = True
                    objective = conditional_objective(values, theta, mean, sd)
                    if not math.isfinite(objective):
                        raise ValueError("Nonfinite summed conditional objective")
                    call["conditional_objective"] = objective
                    report["completed_likelihood_calls"] += 1
                except Exception as error:
                    call["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    call["elapsed_seconds"] = time.perf_counter()-started
                    checkpoint()
                check()
        report["summary"] = summarize_points(report["points"], REPORTED_OBJECTIVE)
        json.dumps(report["summary"], allow_nan=False)
        check()
        complete = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            complete = False
            report["error"] = "RuntimeError: Internal whole-process deadline exceeded before final publication"
        report.update(status="urgency_endpoint_diagnostic_completed" if complete else "technical_endpoint_diagnostic_failure",
            diagnostic_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        if not complete:
            report["summary"] = None
        checkpoint()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_endpoint_diagnostic_failure", diagnostic_complete=False, summary=None,
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
        "attempted_likelihood_calls", "completed_likelihood_calls", "elapsed_seconds", "error") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
