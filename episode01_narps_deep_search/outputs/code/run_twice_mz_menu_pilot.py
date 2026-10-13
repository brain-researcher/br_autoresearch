#!/usr/bin/env python3
"""Complete the fixed rival menu on preserved twice-SESOI MZ index zero.

This pilot reuses the completed M0/MZ panel and freshly fits eight rival halves.
It supplies nine nested diagnostics, never nonnested arbitration or recovery
qualification. Source simulation, optimizer settings and historical files stay
unchanged. Only stdlib dependencies load before the real compute/thread guard.
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
from types import SimpleNamespace

from qualify_assets import require_compute


EPISODE = Path(__file__).resolve().parents[2]
SOURCE_REPORT = EPISODE / "outputs/runs/behavioral_mechanism_kill_test/20261005-MZ-effect-level-representatives-46603942/twice/replicate0/mz_effect_level_recovery.json"
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
REPORT_NAME = "twice_mz_menu_pilot.json"
PHASE = "MZ_twice_SESOI_index0_fixed_menu_pilot"
RIVALS = ("MC", "MV", "MVZ", "MVC")
WORKERS, MAX_SECONDS, SEED = 4, 5400, 1900057
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def load_helpers():
    # Some historical modules set thread variables on import. The caller's real
    # environment has already been checked; imports cannot repair a bad guard.
    import hierarchical_behavior_parallel as parallel
    import run_single_process_recovery as core
    import run_single_process_rivals as shared
    import summarize_mz_fixed_menu as menu
    import summarize_mz_twice_wave as twice
    return SimpleNamespace(parallel=parallel, core=core, shared=shared, menu=menu, twice=twice)


def validate_source(report, truth, people, spec, helpers):
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["optimizer"]["maximum_sweeps"] != 500
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01
            or spec.get("joint_z_c_recovery_passed") is not False):
        raise ValueError("Only the original500 six-model fixed-bound implementation is selected")
    helpers.twice.validate_identity(report, 0, spec)
    helpers.twice.completed_diagnostic(report, spec)
    if set(truth) != set(people) or set(report["participant_scores"]) != set(people):
        raise ValueError("Generating truth, original score and synthetic participant identities differ")
    focal = helpers.twice.levels.focal_truth("twice")
    for person, params in truth.items():
        if (set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values())
                or any(params[name] != value for name, value in focal.items())):
            raise ValueError("The saved truth must be exactly the finite uniform z=.40 MZ cell")
        for name in ("kappa", "log_a", "nondecision_logit"):
            low, high = spec["parameters"][name]["bounds"]
            if not low <= params[name] <= high:
                raise ValueError("Saved nuisance truth is outside unchanged support")
        if report["participant_scores"][person]["version"] != people[person][0]["task_version"]:
            raise ValueError("Saved score and synthetic task-version identities differ")
    if report["simulated_deadline_misses"] != sum(int(row["no_response"]) for rows in people.values() for row in rows):
        raise ValueError("The original deadline-miss count differs from the preserved simulation")


def menu_diagnostics(scores, helpers):
    """Use the qualified exact nine-edge arithmetic without selecting a winner."""
    menu = helpers.menu
    menu.score_panel({"participant_scores": scores}, menu.MODELS)
    contrasts = {f"{model}-{baseline}": menu.nested_contrast(scores, model, baseline)
                 for model, baseline in menu.EDGES}
    eligible = {model: all(contrasts[f"{model}-{baseline}"]["qualifies_nested_comparison"]
                          for baseline in menu.NESTED[model]) for model in menu.MODELS}
    maximal = [model for model in menu.MODELS if eligible[model] and not any(
        eligible[other] and menu.MECHANISMS[model] < menu.MECHANISMS[other] for other in menu.MODELS)]
    return {"nested_comparisons": contrasts, "nested_comparison_eligibility": eligible,
            "eligible_models_diagnostic": [model for model in menu.MODELS if eligible[model]],
            "maximal_eligible_models_diagnostic": maximal,
            "multiple_incomparable_eligible_models": len(maximal) > 1}


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    if os.environ.get("SLURM_CPUS_PER_TASK") != "4" or any(os.environ.get(name) != "1" for name in THREAD_ENV):
        raise ValueError("Require four actual allocated CPUs and all six numerical thread limits explicitly one")
    if (args.source_report.resolve() != SOURCE_REPORT.resolve()
            or args.implementation.resolve() != IMPLEMENTATION.resolve()):
        raise ValueError("Only the fixed completed twice-SESOI index0 source and original implementation are selected")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process internal cap must be within1..5400 seconds")
    output = args.output_dir.resolve()
    if (output.exists() or not output.is_relative_to((EPISODE / "outputs/runs").resolve())
            or output == SOURCE_REPORT.parent.resolve() or SOURCE_REPORT.parent.resolve() in output.parents):
        raise FileExistsError("Use a fresh episode run outside the immutable source attempt")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    output.mkdir(parents=True)

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process menu-pilot deadline exceeded")

    report = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "source_report": str(SOURCE_REPORT), "source_dir": str(SOURCE_REPORT.parent),
        "generating_model": "MZ", "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "replicate_index": 0, "seed": SEED, "selected_replicates": [0],
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "started_utc": datetime.now(timezone.utc).isoformat(), "internal_seconds": args.max_seconds,
        "workers": WORKERS, "allocated_cpus": 4, "thread_environment": {name: os.environ[name] for name in THREAD_ENV},
        "original_maximum_sweeps": 500, "status": "running", "menu_pilot_complete": False,
        "source_baseline_validated": False, "saved_scores_replayed": False, "fresh_fit_calls": 0,
        "fits": [], "completed_rivals": [], "scored_components": [], "fit_budgets": [],
        "active_fit": None, "parallel_progress": None, "parallel_completed_sweeps": [],
        "parallel_failure": None, "parallel_workers_cleaned_up": None,
        "simulation_regenerated": False, "saved_M0_MZ_refitted": False, "saved_fit_parameters_reused": False,
        "observed_choices_or_RTs_used": False, "source_files_modified": False,
        "family_selection": "not_evaluated", "family_recovery_rate": "not_evaluated",
        "confusion_rate": "not_evaluated", "group_recovery": "not_evaluated", "interval_coverage": "not_evaluated",
        "full_gate": "not_evaluated", "full_recovery_pass": None, "episode_terminal": False,
        "full_menu_wave_selected": False, "automatic_retry": False,
        "interpretation": "One preserved twice-SESOI cohort, eight fresh original500 rival half-fits and nine nested diagnostics. No nonnested winner, cell rate, interval, recovery gate or later wave is selected.",
        "cost_accounting": "Historical source process/fit costs are references only; coordinator charges the new allocation once."}

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
        raise RuntimeError(f"Menu pilot interrupted by signal {signum}; unfinished work is technical")

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        checkpoint()
        check()
        helpers = load_helpers()
        spec = json.loads(args.implementation.read_text())
        source = json.loads(args.source_report.read_text())
        truth = json.loads((SOURCE_REPORT.parent / "generating_parameters.json").read_text())
        people = helpers.shared.load_cohort(SOURCE_REPORT.parent / "simulated_events.csv.gz", check)
        validate_source(source, truth, people, spec, helpers)
        report.update(implementation=spec, source_baseline_validated=True, source_finished_utc=source["finished_utc"],
            source_process_elapsed_seconds=source["elapsed_seconds"],
            source_fit_elapsed_seconds=[fit["elapsed_seconds"] for fit in source["fits"]],
            focal_generating_truth=source["focal_generating_truth"], participants=108, runs=432, trials=27648,
            model_menu=list(helpers.menu.MODELS),
            parallel_cleanup_reserve_seconds=helpers.parallel.CLEANUP_SECONDS + 1.)
        check()
        splits = [helpers.shared.load_split(SOURCE_REPORT.parent / "simulated_events.csv.gz", list(train), list(held))
                  for train, held in helpers.core.HALVES]
        scores, discrepancy = helpers.shared.replay_saved_scores(source, splits, spec, check)
        report.update(saved_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
            saved_score_replay_absolute_tolerance=helpers.shared.REPLAY_ABSOLUTE_TOLERANCE,
            participant_scores=deepcopy(scores))
        checkpoint()
        for model in RIVALS:
            for row in scores.values():
                row[model] = 0.
            for direction, ((train, held), (training, scoring)) in enumerate(zip(helpers.core.HALVES, splits)):
                check()
                report.update(active_fit={"model": model, "direction": direction,
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
                    fit = helpers.parallel.fit_partially_pooled_parallel(training, model, "fixed", spec,
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
                helpers.core.validate_fit(fit, model, train, held, spec, people)
                contribution = helpers.shared.score_fit(fit, scoring, spec, check)
                check()
                for person, value in contribution.items():
                    scores[person][model] += value
                report["scored_components"].append({**report["active_fit"], "heldout_trials_per_participant": 128,
                    "per_participant_denominator": 256, "participant_contributions": contribution})
                checkpoint()
            report["completed_rivals"].append(model)
            report["participant_scores"] = {person: {name: row[name] for name in
                ("version", "M0", "MZ", *report["completed_rivals"])} for person, row in scores.items()}
            checkpoint()
        check()
        diagnostics = menu_diagnostics(scores, helpers)
        check()
        report.update(**diagnostics, active_fit=None, total_fresh_fit_seconds=math.fsum(fit["elapsed_seconds"] for fit in report["fits"]))
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
        if not succeeded:
            for field in ("nested_comparisons", "nested_comparison_eligibility", "eligible_models_diagnostic",
                          "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models"):
                report.pop(field, None)
        report.update(status="menu_pilot_completed" if succeeded else "technical_menu_pilot_failure",
                      menu_pilot_complete=succeeded, finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        if succeeded and time.monotonic() >= deadline:
            report.update(status="technical_menu_pilot_failure", menu_pilot_complete=False,
                error="RuntimeError: Internal deadline reached during final publication",
                finished_utc=datetime.now(timezone.utc).isoformat())
            for field in ("nested_comparisons", "nested_comparison_eligibility", "eligible_models_diagnostic",
                          "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models"):
                report.pop(field, None)
            checkpoint()  # One corrective terminal write; external grace remains authoritative.
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("status", "menu_pilot_complete", "fresh_fit_calls",
        "completed_rivals", "maximal_eligible_models_diagnostic", "elapsed_seconds", "error") if key in report},
        indent=2, allow_nan=False), flush=True)
    return 0 if report["menu_pilot_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
