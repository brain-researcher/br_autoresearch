#!/usr/bin/env python3
"""One explicitly dispatched original500 rival pair on saved twice-SESOI MZ.

Capability for indices1..199 is not wave selection. Each process preserves
its returned fits/scored prefix independently and neither retries nor resumes.
No legacy numerical module loads before the actual four-CPU/thread guard.
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


EPISODE = Path(__file__).resolve().parents[2]
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
REPORT_NAME = "twice_mz_rival_component.json"
PHASE = "MZ_twice_SESOI_fixed_rival_component"
RIVALS, THREAD_ENV = pilot.RIVALS, pilot.THREAD_ENV
MAX_SECONDS, WORKERS = 5400, 4


def source_report(index):
    if type(index) is not int or not 1 <= index <= 199:
        raise ValueError("Only fixed saved twice-SESOI indices1 through199 are supported")
    array = 46606307 if index == 1 else 46606413
    return (EPISODE / "outputs/runs/behavioral_mechanism_kill_test" /
            f"20261005-MZ-twice-SESOI-wave-{array}" / f"replicate{index}" / "mz_effect_level_recovery.json")


def validate_source(report, truth, people, spec, index, helpers):
    """Validate the actual indexed twice cell; never rewrite it as index zero."""
    source_report(index)
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["optimizer"]["maximum_sweeps"] != 500
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01
            or spec.get("joint_z_c_recovery_passed") is not False):
        raise ValueError("Only the unchanged original500 six-model fixed-bound implementation is selected")
    helpers.twice.validate_identity(report, index, spec)
    helpers.twice.completed_diagnostic(report, spec)
    if set(truth) != set(people) or set(report["participant_scores"]) != set(people):
        raise ValueError("Generating truth, score and synthetic participant identities differ")
    focal = helpers.twice.levels.focal_truth("twice")
    for person, params in truth.items():
        if (set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values())
                or any(params[name] != value for name, value in focal.items())):
            raise ValueError("The saved truth must be the finite uniform z=.40 MZ cell")
        for name in ("kappa", "log_a", "nondecision_logit"):
            low, high = spec["parameters"][name]["bounds"]
            if not low <= params[name] <= high:
                raise ValueError("Saved nuisance truth is outside unchanged support")
        if report["participant_scores"][person]["version"] != people[person][0]["task_version"]:
            raise ValueError("Saved score and synthetic task-version identities differ")
    if report["simulated_deadline_misses"] != sum(int(row["no_response"]) for rows in people.values() for row in rows):
        raise ValueError("The saved deadline-miss count differs from the preserved simulation")


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise ValueError("Require four actual allocated CPUs and all six numerical thread limits explicitly one")
    source_path = source_report(args.replicate_index)
    if args.model not in RIVALS or args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Select one MC/MV/MVZ/MVC model and the original implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process internal cap must be within1..5400 seconds")
    output = args.output_dir.resolve()
    runs = (EPISODE / "outputs/runs").resolve()
    protected = [source_report(index).parent.parent.resolve() for index in (1, 2)]
    if (output.exists() or not output.is_relative_to(runs)
            or any(output == root or root in output.parents for root in protected)):
        raise FileExistsError("Use a fresh episode run outside both preserved twice-source arrays")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    output.mkdir(parents=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process rival-component deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_report": str(source_path), "source_dir": str(source_path.parent),
        "generating_model": "MZ", "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "replicate_index": args.replicate_index, "seed": 1900057 + 10007 * args.replicate_index,
        "fitted_model": args.model, "selected_replicates": [args.replicate_index],
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "workers": WORKERS, "allocated_cpus": 4, "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "original_maximum_sweeps": 500, "status": "running", "rival_component_complete": False,
        "source_baseline_validated": False, "saved_scores_replayed": False, "fresh_fit_calls": 0,
        "fits": [], "scored_components": [], "scored_primary_directions": [], "completed_rivals": [],
        "fit_budgets": [], "active_fit": None, "parallel_progress": None, "parallel_completed_sweeps": [],
        "parallel_failure": None, "parallel_workers_cleaned_up": None,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False, "saved_fit_parameters_reused": False,
        "observed_choices_or_RTs_used": False, "source_files_modified": False,
        "family_selection": "not_evaluated", "family_recovery_rate": "not_evaluated", "confusion_rate": "not_evaluated",
        "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated", "full_gate": "not_evaluated",
        "full_recovery_pass": None, "episode_terminal": False, "automatic_retry": False, "automatic_resume": False,
        "component_execution_selected_by_coordinator": True, "subsequent_wave_selected": False,
        "interpretation": "One explicitly dispatched twice-SESOI model/index pair. Completed fit and scored prefixes remain available on technical stop; no automatic retry, resume, family selection, interval or recovery verdict.",
        "cost_accounting": "Historical baseline cost is referenced without recharging it; coordinator charges each new allocation once."}

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
        raise RuntimeError(f"Rival component interrupted by signal {signum}; unfinished work is technical")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        helpers = pilot.load_helpers()
        spec = json.loads(args.implementation.read_text())
        source = json.loads(source_path.read_text())
        truth = json.loads((source_path.parent / "generating_parameters.json").read_text())
        events = source_path.parent / "simulated_events.csv.gz"
        people = helpers.shared.load_cohort(events, check)
        validate_source(source, truth, people, spec, args.replicate_index, helpers)
        report.update(implementation=spec, source_baseline_validated=True,
            source_job_id=str(source["job_id"]), source_finished_utc=source["finished_utc"],
            source_process_elapsed_seconds=source["elapsed_seconds"],
            source_fit_elapsed_seconds=[fit["elapsed_seconds"] for fit in source["fits"]],
            focal_generating_truth=source["focal_generating_truth"], participants=108, runs=432, trials=27648,
            parallel_cleanup_reserve_seconds=helpers.parallel.CLEANUP_SECONDS + 1.)
        check()
        splits = [helpers.shared.load_split(events, list(train), list(held)) for train, held in helpers.core.HALVES]
        scores, discrepancy = helpers.shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=helpers.shared.REPLAY_ABSOLUTE_TOLERANCE,
            participant_scores=deepcopy(scores))
        checkpoint()
        for row in scores.values():
            row[args.model] = 0.
        for direction, ((train, held), (training, scoring)) in enumerate(zip(helpers.core.HALVES, splits)):
            check()
            report.update(active_fit={"model": args.model, "direction": direction,
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
                fit = helpers.parallel.fit_partially_pooled_parallel(training, args.model, "fixed", spec,
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
            helpers.core.validate_fit(fit, args.model, train, held, spec, people)
            contribution = helpers.shared.score_fit(fit, scoring, spec, check)
            check()
            for person, value in contribution.items():
                scores[person][args.model] += value
            report["scored_components"].append({**report["active_fit"], "heldout_trials_per_participant": 128,
                "per_participant_denominator": 256, "participant_contributions": contribution})
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            checkpoint()
        # Reuse exact shape/finite score-panel validation; no new contrast or
        # scientific choice is made inside a single component.
        helpers.menu.score_panel({"participant_scores": scores}, ("M0", "MZ", args.model))
        check()
        report.update(participant_scores=scores, completed_rivals=[args.model], active_fit=None,
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
        report.update(status="rival_component_completed" if succeeded else "technical_rival_component_failure",
            rival_component_complete=succeeded, finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        if succeeded and time.monotonic() >= deadline:
            report.update(status="technical_rival_component_failure", rival_component_complete=False,
                error="RuntimeError: Internal deadline reached during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            checkpoint()  # One bounded corrective publication, never a retry.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=RIVALS, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "rival_component_complete", "fitted_model",
        "replicate_index", "seed", "fresh_fit_calls", "elapsed_seconds", "error") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_component_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
