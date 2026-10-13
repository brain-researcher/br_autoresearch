#!/usr/bin/env python3
"""Benchmark a batched urgency candidate against the complete saved scalar pilot.

The48 expensive scalar vectors are reused, not rerun. Agreement at1e-12 is
engineering equivalence only; neither fitting nor solver adoption is selected.
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

import collapsing_ddm_batched as batched
from qualify_assets import require_compute
import run_mz_urgency_cost_precision_pilot as scalar


SOURCE, IMPLEMENTATION = scalar.SOURCE, scalar.IMPLEMENTATION
REFERENCE_JOB = "46607960"
REFERENCE = SOURCE.parent / f"20261005-MZ-urgency-cost-precision-{REFERENCE_JOB}" / scalar.REPORT_NAME
PHASE = "MZ0_urgency_batched_saved_scalar_benchmark"
REPORT_NAME = "mz_urgency_batched_benchmark.json"
MAX_SECONDS, ENGINEERING_TOLERANCE = 600, 1e-12


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def probe_plan(people, splits, spec):
    selected = {version: min(person for person, rows in people.items() if rows[0]["task_version"] == version)
                for version in ("EI", "ER")}
    probes = []
    for (train, held), (training, scoring) in zip(scalar.HALVES, splits):
        for version, person in selected.items():
            rows = training[person]["rows"]
            if len(rows) != 128 or {int(row["run"]) for row in rows} != set(train) or training[person]["version"] != version:
                raise ValueError("Retain each prospectively selected128-trial training half")
            data = scalar.shared.make_data(rows)
            for model in scalar.MODELS:
                names, starts = scalar.initialization_vectors(model, spec)
                for start_index, theta in enumerate(starts):
                    drift, horizon = scalar.drift_and_horizon(theta, names, data)
                    identity = {"participant": person, "task_version": version, "model": model,
                        "training_runs": list(train), "excluded_scoring_runs": list(held), "trials": 128,
                        "trial_slots": [[int(row["run"]), int(row["trial_order"])] for row in rows],
                        "start_index": start_index, "parameter_names": names, "parameters": theta.tolist(),
                        "actual_unique_drift_count": int(len(np.unique(drift))), "decision_horizon_seconds": horizon}
                    probes.append((identity, theta, names, data))
    return selected, probes


def validate_reference(reference, spec, refined, source, primary, selected, plan):
    if (reference.get("analysis_id") != "behavioral_mechanism_kill_test" or reference.get("phase") != scalar.PHASE
            or reference.get("generating_model") != "MZ" or type(reference.get("replicate_index")) is not int
            or reference["replicate_index"] != 0 or reference.get("seed") != scalar.shared.SEED
            or reference.get("rival") != "urgency" or str(reference.get("job_id")) != REFERENCE_JOB
            or reference.get("status") != "urgency_cost_precision_pilot_completed" or reference.get("pilot_complete") is not True
            or reference.get("source_dir") != str(SOURCE) or reference.get("source_report") != str(SOURCE / scalar.SOURCE_REPORT)
            or reference.get("implementation") != spec or reference.get("refined_numerical_implementation") != refined
            or reference.get("source_baseline_validated") is not True or reference.get("saved_primary_scores_replayed") is not True
            or reference.get("source_job_id") != source["job_id"] or reference.get("source_finished_utc") != source["finished_utc"]
            or reference.get("saved_primary_participant_scores") != primary
            or reference.get("selected_probe_participants") != selected
            or [reference.get(field) for field in ("source_participants", "source_runs", "source_trials")] != [108, 432, 27648]
            or [reference.get(field) for field in ("expected_probe_pairs", "completed_probe_pairs", "expected_likelihood_calls", "completed_likelihood_calls")] != [24, 24, 48, 48]
            or not isinstance(reference.get("probes"), list) or len(reference["probes"]) != 24
            or "active_call" in reference or "incomplete_reason" in reference):
        raise ValueError("Only the complete exact48-call scalar pilot on this qualified source may be benchmarked")
    if (any(reference.get(field) != "not_evaluated" for field in scalar.CLAIM_FIELDS)
            or any(reference.get(field) is not False for field in ("simulation_regenerated", "simulation_copied", "observed_choices_or_RTs_used",
                "optimizer_called", "model_fitted", "actual_fitted_support_evaluated", "wave_selected", "automatic_retry", "episode_terminal"))
            or reference.get("full_recovery_pass") is not None
            or reference.get("saved_score_replay_absolute_tolerance") != scalar.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(reference.get("saved_score_replay_max_absolute_discrepancy"))
            or not 0 <= reference["saved_score_replay_max_absolute_discrepancy"] <= scalar.shared.REPLAY_ABSOLUTE_TOLERANCE
            or not finite(reference.get("elapsed_seconds")) or reference["elapsed_seconds"] < 0):
        raise ValueError("Preserve the scalar pilot's exact replay, costs and noninferential claim boundary")
    try:
        began, finished = (datetime.fromisoformat(reference[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid scalar terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Scalar pilot must have valid terminal timestamps") from error
    for saved, (identity, theta, names, data) in zip(reference["probes"], plan):
        if (any(saved.get(key) != value for key, value in identity.items()) or saved.get("pair_complete") is not True
                or not isinstance(saved.get("meshes"), dict) or set(saved["meshes"]) != {"coarse", "refined"}):
            raise ValueError("Scalar probe ordering, actual participants, training rows or initialization changed")
        values = {}
        for mesh, dt, intervals in scalar.MESHES:
            record = saved["meshes"][mesh]
            vector = record.get("trial_log_scores")
            if (not isinstance(vector, list) or len(vector) != 128 or not all(finite(value) for value in vector)
                    or record.get("dt_seconds") != dt or record.get("spatial_intervals") != intervals
                    or record.get("time_steps_per_drift") != math.ceil(identity["decision_horizon_seconds"] / dt)
                    or not finite(record.get("elapsed_seconds")) or record["elapsed_seconds"] <= 0):
                raise ValueError("Each saved scalar resolution requires its finite128-vector and measured positive call cost")
            values[mesh] = np.asarray(vector)
            if record.get("mean_log_score") != float(values[mesh].mean()):
                raise ValueError("Saved scalar mean differs from its trial vector")
        differences = values["refined"] - values["coarse"]
        if saved.get("mesh_difference") != {"direction": "refined_minus_coarse",
                "per_trial_log_score_difference": differences.tolist(),
                "mean_log_score_difference": float(differences.mean()),
                "maximum_absolute_log_score_difference": float(np.max(np.abs(differences)))}:
            raise ValueError("Saved scalar mesh differences do not match both completed vectors")


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Urgency benchmark interrupted by signal {signum}; unfinished comparisons remain incomplete")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the fixed original MZ0 source and unchanged implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The benchmark whole-process cap must be1 through600 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or any(output == protected.resolve() or protected.resolve() in output.parents for protected in (SOURCE, REFERENCE.parent)):
        raise FileExistsError("Use a fresh output outside the preserved source and scalar pilot")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ",
        "replicate_index": 0, "seed": scalar.shared.SEED, "rival": "urgency", "reference_report": str(REFERENCE),
        "reference_job_id": REFERENCE_JOB, "source_dir": str(SOURCE), "source_report": str(SOURCE / scalar.SOURCE_REPORT),
        "source_baseline_validated": False, "saved_primary_scores_replayed": False, "scalar_reference_validated": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "benchmark_complete": False,
        "expected_likelihood_calls": 48, "completed_likelihood_calls": 0, "calls": [],
        "engineering_absolute_tolerance": ENGINEERING_TOLERANCE, "engineering_equivalent_at_1e_minus12": None,
        "maximum_absolute_trial_log_score_discrepancy": None,
        "simulation_regenerated": False, "simulation_copied": False, "observed_choices_or_RTs_used": False,
        "scalar_urgency_calls_repeated": False, "optimizer_called": False, "model_fitted": False,
        "actual_fitted_support_evaluated": False, "solver_adopted": False, "wave_selected": False,
        "automatic_retry": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in scalar.CLAIM_FIELDS},
        "interpretation": "Saved scalar initialization probes versus batched candidate only. The1e-12 threshold is engineering equivalence borrowed explicitly from the existing saved-score replay tolerance, not an inference or fitted-support precision threshold. Single measured call ratios do not guarantee optimizer or whole-fit speed. No fitter, solver adoption or further work is selected.",
        "cost_accounting": "elapsed_seconds charges only this source validation/replay and batched benchmark. Recorded scalar call/process durations are comparison references and are not charged again."}
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        check()
        spec = json.loads(args.implementation.read_text())
        if spec["optimizer"]["maximum_sweeps"] != 500 or spec["urgency_solver"] != {
                "dt_seconds": .002, "spatial_intervals": 200,
                "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}:
            raise ValueError("Preserve the unchanged original500 specification and the two declared meshes")
        refined = deepcopy(spec)
        refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
        report.update(implementation=spec, refined_numerical_implementation=refined)
        reference = json.loads(REFERENCE.read_text())
        if reference.get("pilot_complete") is not True or reference.get("completed_likelihood_calls") != 48:
            raise ValueError("The fixed scalar pilot is incomplete; no complete benchmark may be fabricated")
        check()
        source = json.loads((SOURCE / scalar.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = scalar.shared.load_cohort(events, check)
        scalar.shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"], source_finished_utc=source["finished_utc"],
                      source_participants=108, source_runs=432, source_trials=27648)
        write_report(output, report)
        splits = [scalar.shared.load_split(events, list(train), list(held)) for train, held in scalar.HALVES]
        check()
        primary, discrepancy = scalar.shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True, saved_primary_participant_scores=primary,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=scalar.shared.REPLAY_ABSOLUTE_TOLERANCE)
        selected, plan = probe_plan(people, splits, spec)
        validate_reference(reference, spec, refined, source, primary, selected, plan)
        check()
        report.update(scalar_reference_validated=True, selected_probe_participants=selected,
            scalar_reference_finished_utc=reference["finished_utc"], scalar_process_elapsed_seconds=reference["elapsed_seconds"])
        write_report(output, report)
        for probe_index, (identity, theta, names, data) in enumerate(plan):
            for mesh, dt, intervals in scalar.MESHES:
                check()
                report["active_call"] = {"probe_index": probe_index, "mesh": mesh}
                write_report(output, report)
                call_started = time.perf_counter()
                candidate = np.asarray(batched.trial_likelihood(theta, names, data, spec if mesh == "coarse" else refined))
                elapsed = time.perf_counter() - call_started
                if candidate.shape != (128,) or not np.isfinite(candidate).all() or not elapsed > 0:
                    raise ValueError("Each batched call must retain128 finite scores and a measured positive duration")
                saved = reference["probes"][probe_index]["meshes"][mesh]
                difference = candidate - np.asarray(saved["trial_log_scores"])
                maximum = float(np.max(np.abs(difference)))
                record = {**identity, "probe_index": probe_index, "mesh": mesh, "dt_seconds": dt,
                    "spatial_intervals": intervals, "trial_log_scores": candidate.tolist(),
                    "candidate_mean_log_score": float(candidate.mean()), "reference_mean_log_score": saved["mean_log_score"],
                    "mean_log_score_difference": float(difference.mean()), "maximum_absolute_trial_log_score_discrepancy": maximum,
                    "engineering_equivalent_at_1e_minus12": maximum <= ENGINEERING_TOLERANCE,
                    "scalar_reference_elapsed_seconds": saved["elapsed_seconds"], "candidate_elapsed_seconds": elapsed,
                    "measured_scalar_to_candidate_time_ratio": saved["elapsed_seconds"] / elapsed}
                report["calls"].append(record)
                report["completed_likelihood_calls"] += 1
                report["maximum_absolute_trial_log_score_discrepancy"] = max(row["maximum_absolute_trial_log_score_discrepancy"] for row in report["calls"])
                report.pop("active_call", None)
                write_report(output, report)
                check()
        if report["completed_likelihood_calls"] != 48:
            raise ValueError("The selected48-call benchmark panel is incomplete")
        report.update(status="urgency_batched_benchmark_completed", benchmark_complete=True,
            engineering_equivalent_at_1e_minus12=all(row["engineering_equivalent_at_1e_minus12"] for row in report["calls"]),
            elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report["incomplete_reason"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.update(status="urgency_batched_benchmark_incomplete", benchmark_complete=False,
                engineering_equivalent_at_1e_minus12=None, elapsed_seconds=time.perf_counter() - started,
                finished_utc=datetime.now(timezone.utc).isoformat())
            write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "benchmark_complete", "completed_likelihood_calls",
        "engineering_equivalent_at_1e_minus12", "maximum_absolute_trial_log_score_discrepancy", "elapsed_seconds", "incomplete_reason") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["benchmark_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
