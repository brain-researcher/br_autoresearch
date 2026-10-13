#!/usr/bin/env python3
"""One fresh MVC pair after the diagnosed twice-SESOI index0 exhaustion.

Only a private maximum-sweeps ceiling changes. The failed menu, completed
MC/MV/MVZ prefix and original baseline remain immutable; this is not a wave
dispatcher or a full-menu calculation. Numerical helpers load after the real
four-CPU/thread guard and inside the whole-process timer.
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
import socket
import time

from qualify_assets import require_compute
import run_twice_mz_menu_pilot as pilot


EPISODE, IMPLEMENTATION, SOURCE_REPORT = pilot.EPISODE, pilot.IMPLEMENTATION, pilot.SOURCE_REPORT
FAILED_REPORT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-twice-menu-pilot-46634879-attempt001/twice_mz_menu_pilot.json"
FAILED_JOB = "46634879"
FAILED_FINISHED = "2026-10-05T17:00:28.107639+00:00"
SOURCE_FINISHED = "2026-10-05T10:17:37.205713+00:00"
REPORT_NAME = "twice_mz_mvc_repair_pilot.json"
PHASE = "MZ_twice_SESOI_index0_MVC_runtime1500_repair_pilot"
MODEL, PRESERVED = "MVC", ("MC", "MV", "MVZ")
ORIGINAL_SWEEPS, RUNTIME_SWEEPS, WORKERS, MAX_SECONDS = 500, 1500, 4, 5400
THREAD_ENV = pilot.THREAD_ENV
NONCONVERGENCE = "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"
CLAIMS = ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery",
          "interval_coverage", "full_gate", "full_menu_selection")


def runtime_specification(original):
    if original["optimizer"]["maximum_sweeps"] != ORIGINAL_SWEEPS:
        raise ValueError("The immutable original implementation must retain500 sweeps")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    return runtime


def repair_context(failed, source, people, original, helpers):
    """Validate exact failure and completed prefixes; never reuse failed state."""
    import summarize_twice_mz_menu_wave as collector

    admitted, evidence = collector.validate_rivals(failed, source, 0, original, pilot.RIVALS, is_pilot=True)
    if (str(failed["job_id"]) != FAILED_JOB or failed.get("finished_utc") != FAILED_FINISHED
            or failed.get("status") != "technical_menu_pilot_failure" or failed.get("error") != NONCONVERGENCE
            or failed.get("parallel_workers_cleaned_up") is not True or failed.get("parallel_failure") is not None
            or failed.get("fresh_fit_calls") != 7 or failed.get("completed_rivals") != list(PRESERVED)
            or len(failed["fits"]) != 7 or len(failed["scored_components"]) != 6
            or set(admitted) != set(PRESERVED)
            or str(source["job_id"]) != "46603942" or source["finished_utc"] != SOURCE_FINISHED
            or any(fit["participants"] != sorted(people) for fit in failed["fits"])):
        raise ValueError("Only the exact terminal index0 forward-MVC500 exhaustion is selected")
    fit = failed["fits"][-1]
    history = fit.get("objective_history")
    if (fit.get("model") != MODEL or fit.get("rival") != "fixed" or fit.get("converged") is not False
            or fit.get("training_runs") != [1, 4] or fit.get("scoring_runs") != [2, 3]
            or fit.get("subject_optimizer_failures") != [] or not isinstance(history, list)
            or len(history) != ORIGINAL_SWEEPS
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in history)):
        raise ValueError("Require one finite500-sweep failed forward fit without subject optimizer failures")
    json.dumps(fit, allow_nan=False)
    delta = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if delta <= threshold:
        raise ValueError("The saved last sweep does not demonstrate the diagnosed unchanged-criterion exhaustion")
    return {
        "kind": "maximum_sweeps_only", "classification": "first_half_500_sweep_exhaustion",
        "failed_report": str(FAILED_REPORT), "failed_job_id": FAILED_JOB,
        "failed_finished_utc": FAILED_FINISHED, "failed_error": failed["error"],
        "failed_source_finished_utc": failed["source_finished_utc"],
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in failed["fits"]],
        "previous_MVC_fit": deepcopy(fit),
        "previous_completed_rivals": deepcopy(failed["completed_rivals"]),
        "previous_completed_fits": deepcopy(failed["fits"][:6]),
        "previous_scored_components": deepcopy(failed["scored_components"]),
        "previous_participant_scores": deepcopy(failed["participant_scores"]),
        "previous_terminal_evidence": evidence,
        "final_objective_change": delta, "unchanged_convergence_threshold": threshold,
        "original_maximum_sweeps": ORIGINAL_SWEEPS, "runtime_maximum_sweeps": RUNTIME_SWEEPS,
        "nonconverged_checkpoint_state_reused": False,
        "initialization_of_fresh_fits": "unchanged_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "cost_accounting": "Historical process and included fit durations are references only, not additive new charges.",
    }


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise ValueError("Require four actual allocated CPUs and all six numerical thread limits explicitly one")
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use only the immutable original implementation; source and failed index0 paths are fixed")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process internal cap must be within1..5400 seconds")
    output = args.output_dir.resolve()
    if (output.exists() or not output.is_relative_to((EPISODE / "outputs/runs").resolve())
            or any(output == parent.resolve() or parent.resolve() in output.parents
                   for parent in (SOURCE_REPORT.parent, FAILED_REPORT.parent))):
        raise FileExistsError("Use a fresh episode output outside both immutable historical attempts")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    output.mkdir(parents=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process MVC-repair-pilot deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_report": str(SOURCE_REPORT), "source_dir": str(SOURCE_REPORT.parent),
        "failed_report": str(FAILED_REPORT), "generating_model": "MZ", "fitted_model": MODEL,
        "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "replicate_index": 0, "seed": pilot.SEED, "selected_replicates": [0],
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "workers": WORKERS, "allocated_cpus": 4, "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "original_maximum_sweeps": ORIGINAL_SWEEPS, "runtime_maximum_sweeps": RUNTIME_SWEEPS,
        "status": "running", "mvc_repair_complete": False,
        "source_baseline_validated": False, "repair_context_validated": False, "saved_scores_replayed": False,
        "implementation": None, "runtime_implementation": None, "numerical_repair": None,
        "fresh_fit_calls": 0, "fits": [], "scored_components": [], "scored_primary_directions": [],
        "completed_rivals": [], "fit_budgets": [], "active_fit": None,
        "parallel_progress": None, "parallel_completed_sweeps": [],
        "parallel_failure": None, "parallel_workers_cleaned_up": None,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False, "saved_fit_parameters_reused": False,
        "observed_choices_or_RTs_used": False, "source_files_modified": False,
        **{field: "not_evaluated" for field in CLAIMS}, "full_recovery_pass": None, "episode_terminal": False,
        "full_menu_wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "automatic_resume": False,
        "interpretation": "One diagnosed twice0 MVC-only repair, with fresh prior starts and a private1500 ceiling. Original500 baseline and completed MC/MV/MVZ evidence remain separate. No full-menu, arbitration, rate, interval or recovery verdict.",
        "cost_accounting": "Each historical process and its included fit durations are references, not added charges. elapsed_seconds is this new process only; coordinator accounts each allocation once."}

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(output, report)

    def progress(event):
        check()
        report["parallel_progress"] = deepcopy(event)
        if event["stage"] == "sweep":
            report["parallel_completed_sweeps"].append({**report["active_fit"], **deepcopy(event)})
            if event["sweep"] % 25 == 0 or event.get("converged"):
                checkpoint()

    def interrupted(signum, frame):
        raise RuntimeError(f"MVC repair pilot interrupted by signal {signum}; unfinished work is technical")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        helpers = pilot.load_helpers()
        original = json.loads(args.implementation.read_text())
        runtime = runtime_specification(original)
        report.update(implementation=original, runtime_implementation=runtime)
        source = json.loads(SOURCE_REPORT.read_text())
        truth = json.loads((SOURCE_REPORT.parent / "generating_parameters.json").read_text())
        events = SOURCE_REPORT.parent / "simulated_events.csv.gz"
        people = helpers.shared.load_cohort(events, check)
        pilot.validate_source(source, truth, people, original, helpers)
        report.update(source_baseline_validated=True, source_job_id=str(source["job_id"]),
            source_finished_utc=source["finished_utc"], source_process_elapsed_seconds=source["elapsed_seconds"],
            source_fit_elapsed_seconds=[fit["elapsed_seconds"] for fit in source["fits"]],
            focal_generating_truth=source["focal_generating_truth"], participants=108, runs=432, trials=27648,
            parallel_cleanup_reserve_seconds=helpers.parallel.CLEANUP_SECONDS + 1.)
        check()
        failed = json.loads(FAILED_REPORT.read_text())
        check()
        context = repair_context(failed, source, people, original, helpers)
        check()
        report.update(repair_context_validated=True, numerical_repair=context)
        checkpoint()
        splits = [helpers.shared.load_split(events, list(train), list(held)) for train, held in helpers.core.HALVES]
        scores, discrepancy = helpers.shared.replay_saved_scores(source, splits, original, check)
        report.update(saved_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=helpers.shared.REPLAY_ABSOLUTE_TOLERANCE,
            participant_scores=deepcopy(scores))
        checkpoint()
        for row in scores.values():
            row[MODEL] = 0.
        for direction, ((train, held), (training, scoring)) in enumerate(zip(helpers.core.HALVES, splits)):
            check()
            report.update(active_fit={"model": MODEL, "direction": direction,
                "training_runs": list(train), "scoring_runs": list(held)},
                parallel_progress=None, parallel_workers_cleaned_up=None)
            checkpoint()
            check()
            remaining = deadline - time.monotonic() - report["parallel_cleanup_reserve_seconds"]
            if remaining <= 0:
                raise RuntimeError("Insufficient time remains for bounded participant-worker cleanup")
            report["fit_budgets"].append({**report["active_fit"], "max_seconds": remaining})
            report["fresh_fit_calls"] += 1
            fit_started = time.perf_counter()
            try:
                fit = helpers.parallel.fit_partially_pooled_parallel(training, MODEL, "fixed", runtime,
                    workers=WORKERS, max_seconds=remaining, progress=progress)
            except BaseException as error:
                report["parallel_failure"] = deepcopy(getattr(error, "parallel_progress", None))
                if report["parallel_failure"] is not None:
                    report["parallel_workers_cleaned_up"] = report["parallel_failure"].get("workers_cleaned_up")
                raise
            report["parallel_workers_cleaned_up"] = True
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            checkpoint()
            check()
            helpers.core.validate_fit(fit, MODEL, train, held, runtime, people)
            contribution = helpers.shared.score_fit(fit, scoring, original, check)
            check()
            for person, value in contribution.items():
                scores[person][MODEL] += value
            report["scored_components"].append({**report["active_fit"], "heldout_trials_per_participant": 128,
                "per_participant_denominator": 256, "participant_contributions": contribution})
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            checkpoint()
        helpers.menu.score_panel({"participant_scores": scores}, ("M0", "MZ", MODEL))
        check()
        report.update(participant_scores=scores, completed_rivals=[MODEL], active_fit=None,
            total_fresh_fit_seconds=math.fsum(fit["elapsed_seconds"] for fit in report["fits"]))
        succeeded = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        try:
            check()
        except BaseException as error:
            succeeded = False
            report["error"] = f"{type(error).__name__}: {error}"
        report.update(status="mvc_repair_completed" if succeeded else "technical_mvc_repair_failure",
            mvc_repair_complete=succeeded, finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        if succeeded and time.monotonic() >= deadline:
            report.update(status="technical_mvc_repair_failure", mvc_repair_complete=False,
                error="RuntimeError: Internal deadline reached during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "mvc_repair_complete", "fresh_fit_calls",
        "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["mvc_repair_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
