#!/usr/bin/env python3
"""Two batched A/C resolution calls, preserving the completed coarse anchors."""
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
import run_mz_urgency_crossed_nondecision_diagnostic as crossed


optimization = crossed.optimization
SOURCE, IMPLEMENTATION = crossed.SOURCE, crossed.IMPLEMENTATION
COARSE_REPORT = SOURCE.parent / "20261005-MZ-urgency-crossed-nondecision-46618614-attempt001" / crossed.REPORT_NAME
PHASE = "MZ0_urgency_saved_A_C_two_point_refined_batched_resolution_diagnostic"
REPORT_NAME = "mz_urgency_refined_crossed_diagnostic.json"
MAX_SECONDS = 120
POINTS = ("A", "C")


def validate_coarse(report, spec, source, selected, rows):
    """Bind the completed admitted anchors without reopening their ancestry."""
    identity, _ = optimization.probe_identity(2, selected)
    names, _ = optimization.scalar.initialization_vectors("M0", spec)
    mean = np.array([spec["parameters"][name]["population_center"] for name in names])
    sd = np.exp(np.log([spec["parameters"][name]["individual_scale_center"] for name in names]))
    expected = {**identity, "phase": crossed.PHASE, "job_id": "46618614",
        "status": "crossed_nondecision_diagnostic_completed", "diagnostic_complete": True,
        "source_baseline_validated": True, "saved_primary_scores_replayed": True,
        "benchmark_validated": True, "saved_anchors_validated": True,
        "source_dir": str(SOURCE), "source_job_id": source["job_id"],
        "source_finished_utc": source["finished_utc"], "implementation": spec,
        "trace_report": str(crossed.TRACE), "scalarA_reference": str(crossed.A_REFERENCE),
        "probe_report": str(crossed.PROBE), "benchmark_report": str(optimization.BENCHMARK),
        "parameter_names": names, "conditional_prior_mean": mean.tolist(), "conditional_prior_sd": sd.tolist(),
        "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
        "expected_new_likelihood_calls": 5, "attempted_new_likelihood_calls": 5, "completed_new_likelihood_calls": 5,
        "new_call_order": [{"point": name, "solver": solver} for name, solver in crossed.CALLS],
        "A_reevaluated": False, "optimizer_called": False, "optimizer_settings_changed": False,
        "mesh_changed": False, "heldout_urgency_scores_computed": False, "full_fitter_solver_adopted": False}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Use the completed fixed coarse crossing with this original source, prior and128 slots")
    began, ended = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
    if began.utcoffset() is None or ended.utcoffset() is None or ended < began or len(rows) != 128:
        raise ValueError("Require a terminal coarse result and all128 training rows")
    saved = report["points"]
    if saved["A"]["parameters"] != crossed.endpoint.ENDPOINT:
        raise ValueError("Keep the exact previously selected A endpoint")
    c, _ = crossed.crossed_points(saved["A"]["parameters"], saved["B"]["parameters"], names)
    if saved["C"]["parameters"] != c:
        raise ValueError("C must retain the previously selected one-coordinate crossing")
    points = {}
    for name in POINTS:
        point, call = saved[name], saved[name]["calls"]["batched"]
        if (call.get("returned") is not True or call.get("finite128_vector") is not True
                or len(call["trial_log_scores"]) != 128 or not np.isfinite(call["trial_log_scores"]).all()
                or any(call[key] != value for key, value in crossed.operands(
                    call["trial_log_scores"], point["parameters"], mean, sd).items())):
            raise ValueError("Preserve the finite saved batched anchors and original objective operands")
        points[name] = {"parameters": deepcopy(point["parameters"]),
            "coarse": {"geometry": deepcopy(point["geometry"]), "call": deepcopy(call)},
            "refined": {"geometry": None, "call": None}}
    return points, names, mean, sd


def summarize_refinement(points, mean, sd, slots):
    records = {}
    for name in POINTS:
        for mesh in ("coarse", "refined"):
            entry = points[name][mesh]
            call = entry["call"]
            operands = crossed.operands(call["trial_log_scores"], points[name]["parameters"], mean, sd)
            if any(call[key] != value for key, value in operands.items()):
                raise ValueError("Keep unchanged prior and conditional-objective arithmetic")
            records[name, mesh] = {"evaluation_index": f"{name}_{mesh}",
                "parameters": points[name]["parameters"], "geometry": entry["geometry"],
                "trial_log_scores": call["trial_log_scores"], **operands}
    compare = crossed.trace_analysis.compare_evaluations
    return {"refined_minus_coarse": {name: compare(records[name, "coarse"], records[name, "refined"], slots)
                for name in POINTS},
        "A_to_C": {mesh: compare(records["A", mesh], records["C", mesh], slots)
                for mesh in ("coarse", "refined")},
        "interpretation": "These two saved-state resolution comparisons are descriptive. They do not qualify fitted-support precision, global solver equivalence or optimizer convergence, and do not select a mesh or acceptance threshold. Bin and spatial indices are mesh-relative: different integers across meshes do not themselves identify a discontinuity. Nondecision time changes both lookup geometry and the decision horizon/actual time step."}


def write_report(output, report):
    temporary = output / (REPORT_NAME+".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Refined crossed diagnostic interrupted by signal {signum}")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the unchanged original MZ0 source and implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Bound the entire two-point process to120 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(
            os.environ.get(key) != "1" for key in crossed.trace_analysis.THREAD_ENV):
        raise ValueError("Use one actually allocated CPU and six numerical thread limits of one")
    output = args.output_dir.resolve()
    protected = (SOURCE, COARSE_REPORT.parent, crossed.TRACE.parent, crossed.A_REFERENCE.parent,
                 crossed.PROBE.parent.parent, optimization.BENCHMARK.parent, optimization.benchmark.REFERENCE.parent)
    if output.exists() or any(output == root.resolve() or root.resolve() in output.parents for root in protected):
        raise FileExistsError("Use a fresh output outside all preserved source/reference attempts")
    output.mkdir(parents=True)
    began, deadline = time.perf_counter(), time.monotonic()+args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_dir": str(SOURCE), "coarse_report": str(COARSE_REPORT),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "diagnostic_complete": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False, "coarse_anchors_validated": False,
        "expected_new_likelihood_calls": 2, "attempted_new_likelihood_calls": 0, "completed_new_likelihood_calls": 0,
        "new_call_order": [], "points": {}, "summary": None,
        "new_coarse_urgency_calls": 0, "new_scalar_urgency_calls": 0, "optimizer_called": False,
        "optimizer_settings_changed": False, "simulation_regenerated": False, "observed_choices_or_RTs_used": False,
        "population_hyperparameters_updated": False, "heldout_urgency_scores_computed": False,
        "full_fitter_solver_adopted": False, "fitted_support_precision_qualified": False,
        "mesh_adopted": False, "acceptance_threshold_selected": False, "automatic_retry": False,
        "full_recovery_pass": None, **{field: "not_evaluated" for field in optimization.scalar.CLAIM_FIELDS},
        "cost_accounting": "This process charges original-source replay, anchor admission and two refined batched calls once. Previous coarse/trace/reference costs are historical and are not recharged."}

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
            raise ValueError("Preserve the original500 implementation and existing coarse/refined resolution convention")
        refined = deepcopy(spec)
        refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
        report.update(implementation=spec, refined_numerical_implementation=refined)
        shared = optimization.scalar.shared
        source = json.loads((SOURCE / optimization.scalar.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = shared.load_cohort(events, check)
        shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"], source_finished_utc=source["finished_utc"])
        splits = [shared.load_split(events, list(train), list(held)) for train, held in optimization.scalar.HALVES]
        _, discrepancy = shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=shared.REPLAY_ABSOLUTE_TOLERANCE)
        selected, _ = optimization.benchmark.probe_plan(people, splits, spec)
        identity, half = optimization.probe_identity(2, selected)
        rows = splits[half][0][identity["participant"]]["rows"]
        coarse = json.loads(COARSE_REPORT.read_text())
        points, names, mean, sd = validate_coarse(coarse, spec, source, selected, rows)
        data = shared.make_data(rows)
        for name in POINTS:
            theta = np.asarray(points[name]["parameters"])
            if points[name]["coarse"]["geometry"] != crossed.trace_producer.compact_geometry(theta, names, data, spec):
                raise ValueError("Saved coarse geometry must retain this exact128-row source binding")
            points[name]["refined"]["geometry"] = crossed.trace_producer.compact_geometry(theta, names, data, refined)
        report.update(identity, coarse_anchors_validated=True, coarse_finished_utc=coarse["finished_utc"],
            points=points, parameter_names=names, conditional_prior_mean=mean.tolist(), conditional_prior_sd=sd.tolist(),
            trial_slots=[[int(row["run"]), int(row["trial_order"])] for row in rows])
        check()
        for name in POINTS:
            check()
            point = points[name]
            call = {"new_call": True, "solver": "batched", "returned": False, "finite128_vector": False,
                "trial_log_scores": None, "elapsed_seconds": 0., "nll": None,
                "prior_penalty": None, "conditional_objective": None}
            point["refined"]["call"] = call
            report["attempted_new_likelihood_calls"] += 1
            report["new_call_order"].append(name)
            checkpoint()
            started = time.perf_counter()
            try:
                check()
                theta = np.asarray(point["parameters"])
                values = np.asarray(optimization.batched.trial_likelihood(theta, names, data, refined))
                call["returned"] = True
                if values.ndim == 1:
                    call["trial_log_scores"] = [float(value) if np.isfinite(value) else None for value in values]
                if values.shape != (128,) or not np.isfinite(values).all():
                    raise ValueError("Retain exactly128 finite refined scores; no row deletion")
                operands = crossed.operands(values, theta, mean, sd)
                if not all(math.isfinite(value) for value in operands.values()):
                    raise ValueError("Refined conditional operands must be finite")
                call.update(operands, finite128_vector=True)
                report["completed_new_likelihood_calls"] += 1
            except Exception as error:
                call["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                call["elapsed_seconds"] = time.perf_counter()-started
                checkpoint()
            check()
        report["summary"] = summarize_refinement(points, mean, sd, report["trial_slots"])
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
        report.update(status="refined_crossed_diagnostic_completed" if complete else "technical_refined_crossed_failure",
            diagnostic_complete=complete, finished_utc=datetime.now(timezone.utc).isoformat())
        if not complete:
            report["summary"] = None
        checkpoint()
        if complete and time.monotonic() >= deadline:
            report.update(status="technical_refined_crossed_failure", diagnostic_complete=False, summary=None,
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
