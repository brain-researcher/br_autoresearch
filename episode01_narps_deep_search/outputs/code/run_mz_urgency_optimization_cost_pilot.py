#!/usr/bin/env python3
"""One of eight fixed conditional first-sweep urgency cost probes on MZ0.

This measures the original three-start subject optimization at its unchanged
initial version prior. No population update or held-out urgency score occurs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

import collapsing_ddm_batched as batched
from qualify_assets import require_compute
import run_mz_urgency_batched_benchmark as benchmark
import run_mz_urgency_cost_precision_pilot as scalar
from urgency_conditional_optimization import optimize_subject_block


SOURCE, IMPLEMENTATION = scalar.SOURCE, scalar.IMPLEMENTATION
BENCHMARK = SOURCE.parent / "20261005-MZ-urgency-batched-benchmark-46607960" / benchmark.REPORT_NAME
PHASE = "MZ0_urgency_conditional_first_sweep_optimization_cost_pilot"
REPORT_NAME = "mz_urgency_optimization_cost_pilot.json"
MAX_SECONDS = 3000


def probe_identity(index, selected):
    if type(index) is not int or not 0 <= index < 8:
        raise ValueError("Only the eight fixed half/version/model probes are selected")
    half, remainder = divmod(index, 4)
    version, model = ("EI", "ER")[remainder // 2], scalar.MODELS[remainder % 2]
    train, held = scalar.HALVES[half]
    return {"probe_index": index, "participant": selected[version], "task_version": version,
        "model": model, "training_runs": list(train), "excluded_scoring_runs": list(held),
        "training_trials": 128}, half


def validate_benchmark(report, reference, spec, source, primary, selected, plan):
    """Bind the completed exact48-call comparison to this full source replay."""
    refined = deepcopy(spec)
    refined["urgency_solver"].update(dt_seconds=.001, spatial_intervals=400)
    benchmark.validate_reference(reference, spec, refined, source, primary, selected, plan)
    expected = {"analysis_id": "behavioral_mechanism_kill_test", "phase": benchmark.PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": scalar.shared.SEED,
        "rival": "urgency", "source_dir": str(SOURCE), "source_report": str(SOURCE / scalar.SOURCE_REPORT),
        "reference_report": str(benchmark.REFERENCE), "reference_job_id": benchmark.REFERENCE_JOB,
        "status": "urgency_batched_benchmark_completed", "benchmark_complete": True,
        "expected_likelihood_calls": 48, "completed_likelihood_calls": 48,
        "engineering_absolute_tolerance": benchmark.ENGINEERING_TOLERANCE,
        "engineering_equivalent_at_1e_minus12": True,
        "maximum_absolute_trial_log_score_discrepancy": 0.,
        "source_baseline_validated": True, "saved_primary_scores_replayed": True,
        "scalar_reference_validated": True, "source_participants": 108, "source_runs": 432,
        "source_trials": 27648, "implementation": spec, "refined_numerical_implementation": refined,
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "saved_primary_participant_scores": primary, "selected_probe_participants": selected,
        "scalar_reference_finished_utc": reference["finished_utc"],
        "saved_score_replay_absolute_tolerance": scalar.shared.REPLAY_ABSOLUTE_TOLERANCE}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("The fixed complete benchmark does not match this source and implementation")
    if (str(report.get("job_id")) != benchmark.REFERENCE_JOB
            or any(key in report for key in ("active_call", "incomplete_reason"))
            or not isinstance(report.get("calls"), list) or len(report["calls"]) != 48
            or not benchmark.finite(report.get("saved_score_replay_max_absolute_discrepancy"))
            or not 0 <= report["saved_score_replay_max_absolute_discrepancy"] <= scalar.shared.REPLAY_ABSOLUTE_TOLERANCE
            or any(report.get(key) is not False for key in ("optimizer_called", "model_fitted",
                "observed_choices_or_RTs_used", "simulation_regenerated", "simulation_copied", "solver_adopted"))):
        raise ValueError("Require the completed, noninferential benchmark with its original replay")
    started, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
    if started.utcoffset() is None or finished.utcoffset() is None or finished < started:
        raise ValueError("Benchmark terminal timestamps are invalid")
    for index, (identity, theta, names, data) in enumerate(plan):
        for mesh_index, (mesh, dt, intervals) in enumerate(scalar.MESHES):
            call = report["calls"][2*index + mesh_index]
            values = call.get("trial_log_scores")
            saved = reference["probes"][index]["meshes"][mesh]["trial_log_scores"]
            if (any(call.get(key) != value for key, value in identity.items())
                    or call.get("probe_index") != index or call.get("mesh") != mesh
                    or call.get("dt_seconds") != dt or call.get("spatial_intervals") != intervals
                    or call.get("engineering_equivalent_at_1e_minus12") is not True
                    or call.get("maximum_absolute_trial_log_score_discrepancy") != 0.
                    or call.get("mean_log_score_difference") != 0.
                    or not isinstance(values, list) or len(values) != 128
                    or not all(benchmark.finite(value) for value in values) or values != saved):
                raise ValueError("Require all48 exact scalar/batched vectors in their original order")


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Conditional urgency cost probe interrupted by signal {signum}")


def run(args):
    require_compute()
    if args.source_dir.resolve() != SOURCE.resolve() or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the original fixed MZ0 source and unchanged implementation")
    if type(args.probe_index) is not int or not 0 <= args.probe_index < 8:
        raise ValueError("probe-index must be one of0 through7")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole process, including source replay, is capped at3000 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or any(output == path.resolve() or path.resolve() in output.parents
            for path in (SOURCE, BENCHMARK.parent, benchmark.REFERENCE.parent)):
        raise FileExistsError("Use a fresh output outside all preserved source and benchmark attempts")
    output.mkdir(parents=True)
    began, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": scalar.shared.SEED, "rival": "urgency",
        "probe_index": args.probe_index, "source_dir": str(SOURCE),
        "source_report": str(SOURCE / scalar.SOURCE_REPORT), "benchmark_report": str(BENCHMARK),
        "scalar_reference_report": str(benchmark.REFERENCE), "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "status": "running", "probe_complete": False, "conditional_optimization_complete": False,
        "source_baseline_validated": False, "saved_primary_scores_replayed": False,
        "benchmark_validated": False, "expected_starts": 3, "completed_starts": 0,
        "attempts": [], "best_start": None, "optimizer_called": False,
        "likelihood_calls_attempted": 0, "likelihood_calls_returned": 0, "likelihood_seconds": 0.,
        "simulation_regenerated": False, "simulation_copied": False, "observed_choices_or_RTs_used": False,
        "population_hyperparameters_updated": False, "full_cohort_model_fitted": False,
        "heldout_urgency_scores_computed": False, "actual_fitted_support_precision_evaluated": False,
        "batched_solver_adoption_scope": "this_conditional_cost_pilot_only", "full_fitter_solver_adopted": False,
        "wave_selected": False, "automatic_retry": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in scalar.CLAIM_FIELDS},
        "interpretation": "Three original conditional first-sweep starts at the fixed initial version prior. Completion means all three optimizer calls returned; best.success is separate. A timeout is censored resource evidence. No two-person population fit, held-out urgency inference or full fitted-support precision qualification.",
        "cost_accounting": "Source validation/replay and this task are timed once. Root charges the parent allocation, not parent plus children. Prior scalar/benchmark durations are not new charges."}

    def checkpoint():
        report["completed_starts"] = sum(row["returned"] for row in report["attempts"])
        report["optimizer_called"] = any(row["optimizer_entered"] for row in report["attempts"])
        for key in ("likelihood_calls_attempted", "likelihood_calls_returned", "likelihood_seconds"):
            report[key] = sum(row[key] for row in report["attempts"])
        report["elapsed_seconds"] = time.perf_counter() - began
        write_report(output, report)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    successful = False
    try:
        checkpoint()
        check()
        spec = json.loads(args.implementation.read_text())
        if (spec["optimizer"]["maximum_sweeps"] != 500
                or spec["parameters"]["urgency"]["bounds"] != [0., 2.]
                or spec["urgency_solver"] != {"dt_seconds": .002, "spatial_intervals": 200,
                    "required_precision_check": "repeat_at_dt_0.001_and_400_intervals_at_fitted_support_before_rival_inference"}):
            raise ValueError("Keep the original implementation, supports, optimizer and coarse mesh")
        report["implementation"] = spec
        source = json.loads((SOURCE / scalar.SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = scalar.shared.load_cohort(events, check)
        scalar.shared.validate_source(source, truth, people, spec)
        check()
        report.update(source_baseline_validated=True, source_job_id=source["job_id"],
            source_finished_utc=source["finished_utc"], source_participants=108, source_runs=432, source_trials=27648)
        splits = [scalar.shared.load_split(events, list(train), list(held)) for train, held in scalar.HALVES]
        primary, discrepancy = scalar.shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_primary_scores_replayed=True,
            saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=scalar.shared.REPLAY_ABSOLUTE_TOLERANCE)
        selected, plan = benchmark.probe_plan(people, splits, spec)
        reference = json.loads(benchmark.REFERENCE.read_text())
        completed_benchmark = json.loads(BENCHMARK.read_text())
        validate_benchmark(completed_benchmark, reference, spec, source, primary, selected, plan)
        check()
        identity, half = probe_identity(args.probe_index, selected)
        report.update(identity, benchmark_validated=True, benchmark_finished_utc=completed_benchmark["finished_utc"],
            selected_probe_participants=selected)
        rows = splits[half][0][identity["participant"]]["rows"]
        data = scalar.shared.make_data(rows)
        names, starts = scalar.initialization_vectors(identity["model"], spec)
        priors = [spec["parameters"][name] for name in names]
        centers = np.array([prior["population_center"] for prior in priors])
        initial_sd = np.array([prior["individual_scale_center"] for prior in priors])
        sd = np.exp(np.log(initial_sd))
        report.update(parameter_names=names, conditional_prior_mean=centers.tolist(),
            conditional_prior_sd=sd.tolist(), original_initial_sd=initial_sd.tolist(),
            trial_slots=[[int(row["run"]), int(row["trial_order"])] for row in rows])
        checkpoint()
        report["best_start"] = optimize_subject_block(data, names, spec, centers, sd, starts,
            likelihood=batched.trial_likelihood, attempts=report["attempts"], check=check, checkpoint=checkpoint)
        check()
        successful = True
    except Exception as error:
        report["incomplete_reason"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if time.monotonic() >= deadline:
            successful = False
            report["incomplete_reason"] = "RuntimeError: Internal whole-process deadline exceeded before final publication"
        returned = sum(row["returned"] for row in report["attempts"])
        report.update(probe_complete=returned == 3,
            conditional_optimization_complete=successful and returned == 3,
            status="urgency_conditional_cost_probe_completed" if successful else "urgency_conditional_cost_probe_incomplete",
            finished_utc=datetime.now(timezone.utc).isoformat())
        if not successful:
            report["best_start"] = None
        checkpoint()
        # Final JSON publication belongs to the same process envelope. A
        # bounded corrective write preserves a truthful terminal record if
        # that first publication itself crosses the deadline; never retry it.
        if successful and time.monotonic() >= deadline:
            report.update(status="urgency_conditional_cost_probe_incomplete",
                conditional_optimization_complete=False, best_start=None,
                incomplete_reason="RuntimeError: Internal whole-process deadline exceeded during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-index", type=int, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "probe_index", "status", "probe_complete",
        "conditional_optimization_complete", "completed_starts", "best_start", "likelihood_calls_attempted",
        "likelihood_seconds", "elapsed_seconds", "incomplete_reason") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["conditional_optimization_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
