#!/usr/bin/env python3
"""One explicitly selected saved-MZ MVC numerical repair; no retry wave.

Accept only diagnosed 500-sweep exhaustion from array46572753. Freshly fit
failed halves with the unchanged fitter and a runtime-only 1500-sweep ceiling.
For second-half exhaustion, retain the validated original first fit and rescore
it on the same cohort. Original, pilot42 and shared code remain unchanged.
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

from qualify_assets import require_compute
import run_mz_saved_replicate_rival as saved
import summarize_mz_menu_wave as wave


MODEL = "MVC"
EXCLUDED = frozenset((1, 42, 92, 105))
FAILED_ROOT = saved.RUNS_ROOT / "20261005-MZ-MVC-saved-array-46572753"
REPORT_NAME = "mz_mvc_convergence_repair.json"
PHASE = "MZ_SESOI_saved_MVC_runtime1500_convergence_repair"
ORIGINAL_SWEEPS, RUNTIME_SWEEPS = 500, 1500
MAX_SECONDS, REUSE_MAX_SECONDS = 12600, 6600
QUALIFYING_REPORT = saved.RUNS_ROOT / "20261005-MZ42-MVC-reverse-completion-46587905/mz_mvc_reverse_completion.json"
CLAIM_FIELDS = ("family_selection", "confusion_rate", "group_recovery", "interval_coverage",
                "full_gate", "full_menu_false_selection", "false_selection_rate", "early_kill")
FIRST = "first_half_500_sweep_exhaustion"
SECOND = "second_half_500_sweep_exhaustion"


def failed_path(index):
    saved.source_directory(index)
    if index in EXCLUDED:
        raise ValueError("Held index1, completed42 and successful original MVC92/105 cannot enter this repair")
    return FAILED_ROOT / f"replicate{index}/mz_saved_replicate_rival.json"


def repair_context(index, original, failed):
    """Reuse the tested inventory diagnosis; never infer eligibility from status alone."""
    path = failed_path(index)
    diagnosis = wave.technical_mvc_unknown(failed, index, original, path)["technical_cause"]
    if diagnosis["classification"] not in (FIRST, SECOND):
        raise ValueError("Only evidenced original first/second-half 500-sweep exhaustion is repairable")
    wave.saved.completed_support(failed)
    if (any(failed.get(field) != "not_evaluated" for field in CLAIM_FIELDS[:5])
            or failed.get("full_recovery_pass") is not None or failed.get("episode_terminal") is not False
            or any(failed.get(field) is not False for field in
                   ("simulation_regenerated", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used"))
            or failed.get("diagnostics") != {}):
        raise ValueError("The failed attempt must retain its original diagnostic-only boundary")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    reuse = diagnosis["classification"] == SECOND
    metadata = {
        "kind": "maximum_sweeps_only", "classification": diagnosis["classification"],
        "original_maximum_sweeps": ORIGINAL_SWEEPS, "runtime_maximum_sweeps": RUNTIME_SWEEPS,
        "failed_report": str(path), "failed_error": failed["error"],
        "failed_started_utc": failed["started_utc"], "failed_finished_utc": failed["finished_utc"],
        "failed_source_finished_utc": failed.get("source_finished_utc"),
        "technical_cause": deepcopy(diagnosis),
        "previous_process_elapsed_seconds": failed["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in failed["fits"]],
        "nonconverged_checkpoint_state_reused": False,
        "initialization_of_fresh_fits": "unchanged_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "qualifying_MZ42_report": str(QUALIFYING_REPORT),
        "qualification_boundary": "Coordinator records representative success and selects explicit terminal indices; this runner does not reread the pilot or dispatch a wave",
        "directions": [{"training_runs": list(train), "scoring_runs": list(held),
                        "fit_execution": "reused_validated_original_fit" if reuse and half == 0 else "fresh_fit",
                        "fit_maximum_sweeps": ORIGINAL_SWEEPS if reuse and half == 0 else RUNTIME_SWEEPS}
                       for half, (train, held) in enumerate(saved.HALVES)],
        "source_and_preserved_fit_validated_against_actual_cohort": False,
        "reused_first_fit_original_elapsed_seconds": failed["fits"][0]["elapsed_seconds"] if reuse else None,
        "reused_first_fit_materialization_elapsed_seconds": None,
        "reused_first_fit_rescore_elapsed_seconds": None,
        "reused_first_fit_rescored": False,
        "reused_MVC_score_replay": "not_available_original_participant_scores_were_discarded" if reuse else "not_applicable",
        "cost_accounting": "prior_process_plus_new_process_only; preserved_first_fit_cost_is_already_in_prior_process",
    }
    return runtime, metadata


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def run(args):
    require_compute()
    path = failed_path(args.replicate_index)
    source_dir = saved.source_directory(args.replicate_index)
    if args.implementation.resolve() != saved.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be 1 through12600 seconds; reuse-first is at most6600")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or any(output == protected.resolve() or protected.resolve() in output.parents
                             for protected in (FAILED_ROOT, source_dir, QUALIFYING_REPORT.parent)):
        raise FileExistsError("Use a fresh output outside the preserved source and failed array")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ",
        "fitted_model": MODEL, "replicate_index": args.replicate_index, "seed": saved.cell_seed("MZ", args.replicate_index),
        "source_dir": str(source_dir), "source_report": "single_process_recovery.json", "failed_report": str(path),
        "source_job_id": None, "source_baseline_validated": False, "repair_context_validated": False,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "rival_diagnostics_complete": False,
        "saved_scores_replayed": False, "simulation_regenerated": False, "simulation_copied": False,
        "saved_M0_MZ_refitted": False, "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        "selected_replicates": [args.replicate_index], "selected_models": [MODEL], "fixed_cohort_count": 200,
        "component_scope": "single_MVC_numerical_repair", "wave_selected": False,
        "retry_wave_selected": False, "automatic_retry": False, "full_menu_wave_selected": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS}, "full_recovery_pass": None, "episode_terminal": False,
        "numerical_repair": {"kind": "maximum_sweeps_only", "original_maximum_sweeps": ORIGINAL_SWEEPS,
            "runtime_maximum_sweeps": RUNTIME_SWEEPS, "nonconverged_checkpoint_state_reused": False,
            "source_and_preserved_fit_validated_against_actual_cohort": False},
        "interpretation": "Only diagnosed original500 MVC exhaustion is repaired. Preserve any validated original first fit and rescore it; freshly fit failed halves under runtime1500 with unchanged starts and first-convergence rule. No family, full-menu, rate, interval or gate verdict.",
        "process_state_boundary": "The coordinator separately establishes terminal sources and selects each execution; this runner never polls, retries or dispatches.",
        "fits": [], "scored_primary_directions": [], "completed_rivals": [], "diagnostics": {},
    }
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, saved.interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    metadata = None

    def checkpoint():
        if metadata is not None and "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        write_report(output, report)

    try:
        checkpoint()
        check()
        original = json.loads(args.implementation.read_text())
        failed = json.loads(path.read_text())
        runtime, metadata = repair_context(args.replicate_index, original, failed)
        report.update(implementation=original, runtime_implementation=runtime, numerical_repair=metadata,
                      repair_context_validated=True)
        reuse = metadata["classification"] == SECOND
        if reuse and args.max_seconds > REUSE_MAX_SECONDS:
            raise ValueError("A reuse-first repair is bounded at6600 seconds")
        checkpoint()
        source = json.loads((source_dir / "single_process_recovery.json").read_text())
        truth = json.loads((source_dir / "generating_parameters.json").read_text())
        events = source_dir / "simulated_events.csv.gz"
        people = saved.shared.load_cohort(events, check)
        saved.validate_source(source, truth, people, original, args.replicate_index)
        if failed.get("source_finished_utc") != source["finished_utc"]:
            raise ValueError("Failed attempt source completion differs from the revalidated saved cohort")
        if any(fit.get("participants") != sorted(people) for fit in failed["fits"]):
            raise ValueError("Failed-attempt fit participants differ from the actual saved cohort")
        if reuse:
            saved.validate_fit(failed["fits"][0], MODEL, *saved.HALVES[0], original, people)
        check()
        metadata["source_and_preserved_fit_validated_against_actual_cohort"] = True
        report.update(source_baseline_validated=True, source_job_id=source.get("job_id"),
                      source_finished_utc=source["finished_utc"], participants=108, runs=432, trials=27648)
        checkpoint()
        splits = [saved.load_split(events, list(train), list(held)) for train, held in saved.HALVES]
        check()
        scores, discrepancy = saved.shared.replay_saved_scores(source, splits, original, check)
        report.update(saved_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=saved.shared.REPLAY_ABSOLUTE_TOLERANCE)
        checkpoint()
        for row in scores.values():
            row[MODEL] = 0.
        for half, ((train, held), (training, scoring)) in enumerate(zip(saved.HALVES, splits)):
            check()
            preserved = reuse and half == 0
            fit_started = time.perf_counter()
            if preserved:
                fit = deepcopy(failed["fits"][0])
                fit.update(repair_execution="reused_validated_original_fit", fit_maximum_sweeps=ORIGINAL_SWEEPS,
                           original_fit_elapsed_seconds=failed["fits"][0]["elapsed_seconds"])
            else:
                fit = saved.fit_partially_pooled(training, MODEL, "fixed", runtime)
                fit.update(repair_execution="fresh_fit", fit_maximum_sweeps=RUNTIME_SWEEPS)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            if preserved:
                metadata["reused_first_fit_materialization_elapsed_seconds"] = fit["elapsed_seconds"]
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            checkpoint()
            check()
            saved.validate_fit(fit, MODEL, train, held, original if preserved else runtime, people)
            score_started = time.perf_counter()
            try:
                for person, value in saved.shared.score_fit(fit, scoring, original, check).items():
                    scores[person][MODEL] += value
                if preserved:
                    metadata["reused_first_fit_rescored"] = True
            finally:
                if preserved:
                    metadata["reused_first_fit_rescore_elapsed_seconds"] = time.perf_counter() - score_started
            check()
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            checkpoint()
            print(json.dumps({"model": MODEL, "replicate_index": args.replicate_index,
                              "training_runs": list(train), "fit_completed": True,
                              "repair_execution": fit["repair_execution"], "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        diagnostics = saved.paired_diagnostics(scores, MODEL)
        check()
        report.update(participant_scores=scores, diagnostics={MODEL: diagnostics}, completed_rivals=[MODEL],
                      status="rival_diagnostics_completed", rival_diagnostics_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_rival_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if not succeeded:
            report.pop("participant_scores", None)
            report.update(status="technical_rival_failure", rival_diagnostics_complete=False,
                          completed_rivals=[], diagnostics={}, elapsed_seconds=time.perf_counter() - started,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            # Success finalization was inside the timer. Only technical report
            # closure remains after cancelling it, bounded by the wrapper grace.
            checkpoint()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in
                     ("phase", "status", "replicate_index", "rival_diagnostics_complete", "numerical_repair",
                      "elapsed_seconds", "cumulative_process_elapsed_seconds", "error") if key in report},
                     indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
