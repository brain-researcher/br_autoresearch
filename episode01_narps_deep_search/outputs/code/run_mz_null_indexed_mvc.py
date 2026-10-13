#!/usr/bin/env python3
"""One saved indexed M0-null MVC component, with two fresh runtime1000 fits.

Original500 baselines and their exact saved scores remain unchanged. Null35
uses its separately validated baseline completion, with truth/events still in
the original directory. This bounded runner changes only the MVC sweep ceiling
and process cap; it selects no family, full menu, scientific rate or retry.
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
import run_mz_null_rival as null
import run_mz_null_indexed_simple_rival as indexed_source
import run_mz_null_completed_baseline_rival as completed35


MODEL = "MVC"
ORIGINAL_SWEEPS, RUNTIME_SWEEPS, MAX_SECONDS = 500, 1000, 8400
IMPLEMENTATION = null.IMPLEMENTATION
HALVES = null.HALVES
PHASE = "M0_null_saved_indexed_MVC_runtime1000_diagnostics"
REPORT_NAME = "mz_null_indexed_mvc.json"
CLAIM_FIELDS = (*null.CLAIM_FIELDS, "early_kill")
QUALIFYING_NULL0_REPORT = indexed_source.baseline.RUNS / "20261005-M0-null-MVC-convergence-46582236/mz_null_mvc_convergence_repair.json"
PRESERVED_DIRECTORIES = (*completed35.PRESERVED_DIRECTORIES, QUALIFYING_NULL0_REPORT.parent)

# Scientific functions remain unchanged. There is no source rebinding or
# generated modification of null.run's original5400-second implementation.
fit_partially_pooled = null.fit_partially_pooled
load_split = null.load_split
validate_fit = null.validate_fit


def source_paths(index):
    directory = indexed_source.source_directory(index)
    report = completed35.COMPLETION_REPORT if index == 35 else directory / indexed_source.indexed.REPORT_NAME
    return directory, report


def runtime_specification(original):
    if (original.get("policy_version") != 6 or original.get("state") != "recorded_before_model_scoring"
            or original["optimizer"]["maximum_sweeps"] != ORIGINAL_SWEEPS
            or original["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or original["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or original["contaminant"]["fraction"] != .02 or original["contaminant"]["miss_probability"] != .01):
        raise ValueError("Preserve the original500 implementation and fixed200 null cohort plan")
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = RUNTIME_SWEEPS
    return runtime


def validate_source(report, truth, people, index, original):
    if index == 35:
        failed = json.loads(completed35.FAILED_REPORT.read_text())
        return completed35.validate_completed_baseline(report, failed, truth, people, original)
    indexed_source.baseline.completed_diagnostic(report, truth, people, index, original)
    return None


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Indexed null MVC interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    directory, source_path = source_paths(args.replicate_index)
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("The whole-process cap must be 1 through8400 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Exactly one allocated CPU and all four numerical thread limits of one are required")
    output = args.output_dir.resolve()
    if output.exists() or any(output == path.resolve() or path.resolve() in output.parents for path in PRESERVED_DIRECTORIES):
        raise FileExistsError("Use a fresh output outside all preserved source arrays, completions and pilots")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "M0", "fitted_model": MODEL, "replicate_index": args.replicate_index,
        "seed": indexed_source.indexed.null_seed(args.replicate_index),
        "source_dir": str(directory), "source_report": str(source_path), "source_job_id": None,
        "source_baseline_validated": False, "baseline_lineage": None,
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "rival_diagnostics_complete": False,
        "saved_scores_replayed": False, "simulation_regenerated": False, "simulation_copied": False,
        "saved_M0_MZ_refitted": False, "observed_choices_or_RTs_used": False, "old_synthetic_responses_used": False,
        "selected_replicates": [args.replicate_index], "selected_models": [MODEL], "fixed_cohort_count": 200,
        "wave_selected": True, "wave_scope": "MVC_component_only", "component_scope": "MVC_only",
        "full_menu_wave_selected": False, "retry_selected": False, "automatic_retry": False,
        **{field: "not_evaluated" for field in CLAIM_FIELDS}, "full_recovery_pass": None, "episode_terminal": False,
        "numerical_override": {
            "kind": "maximum_sweeps_only", "original_maximum_sweeps": ORIGINAL_SWEEPS,
            "runtime_maximum_sweeps": RUNTIME_SWEEPS, "fresh_fit_directions": [
                {"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES],
            "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
            "nonconverged_checkpoint_state_reused": False,
            "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
            "qualifying_null0_report": str(QUALIFYING_NULL0_REPORT),
            "qualification_boundary": "Coordinator records terminal representative success; this runner does not reread or refit that pilot",
        },
        "interpretation": "One indexed M0-null MVC component with both directions freshly fitted under runtime1000. Original500 M0/MZ scores are replayed and retained. No full-menu selection, family, false-selection rate, interval, gate or early-kill verdict is computed.",
        "process_state_boundary": "The coordinator separately establishes terminal sources and selects each execution; report fields are not scheduler evidence.",
        "cost_accounting": "elapsed_seconds is this MVC process only; baseline_lineage references shared prior costs without adding them to this component",
        "fits": [], "scored_primary_directions": [], "completed_rivals": [], "diagnostics": {},
    }
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    try:
        write_report(output, report)
        check()
        original = json.loads(args.implementation.read_text())
        runtime = runtime_specification(original)
        report.update(implementation=original, runtime_implementation=runtime)
        source = json.loads(source_path.read_text())
        truth = json.loads((directory / "generating_parameters.json").read_text())
        events = directory / "simulated_events.csv.gz"
        people = null.shared.load_cohort(events, check)
        check()
        lineage = validate_source(source, truth, people, args.replicate_index, original)
        check()
        report.update(source_baseline_validated=True, source_job_id=str(source["job_id"]),
                      source_finished_utc=source["finished_utc"], baseline_lineage=lineage,
                      participants=108, runs=432, trials=27648)
        write_report(output, report)
        splits = [load_split(events, list(train), list(held)) for train, held in HALVES]
        check()
        scores, discrepancy = null.shared.replay_saved_scores(source, splits, original, check)
        report.update(saved_scores_replayed=True, saved_score_replay_max_absolute_discrepancy=discrepancy,
                      saved_score_replay_absolute_tolerance=null.shared.REPLAY_ABSOLUTE_TOLERANCE)
        write_report(output, report)
        for row in scores.values():
            row[MODEL] = 0.
        for (train, held), (training, scoring) in zip(HALVES, splits):
            check()
            fit_started = time.perf_counter()
            fit = fit_partially_pooled(training, MODEL, "fixed", runtime)
            fit.update(training_runs=list(train), scoring_runs=list(held), elapsed_seconds=time.perf_counter() - fit_started)
            json.dumps(fit, allow_nan=False)
            report["fits"].append(fit)
            write_report(output, report)
            check()
            validate_fit(fit, MODEL, train, held, runtime, people)
            # Scoring is unchanged; optimizer ceilings have no likelihood role.
            for person, value in null.shared.score_fit(fit, scoring, original, check).items():
                scores[person][MODEL] += value
            check()
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
            print(json.dumps({"model": MODEL, "replicate_index": args.replicate_index,
                              "training_runs": list(train), "fit_completed": True,
                              "elapsed_seconds": fit["elapsed_seconds"]}), flush=True)
        null.score_panel({"participant_scores": scores}, ("M0", "MZ", MODEL))
        diagnostics = null.paired_diagnostics(scores, MODEL)
        check()
        report.update(participant_scores=scores, diagnostics={MODEL: diagnostics}, completed_rivals=[MODEL],
                      status="rival_diagnostics_completed", rival_diagnostics_complete=True,
                      elapsed_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(output, report)
        # Successful finalization is inside the same deadline as source reads,
        # replay and fitting; a final-write overrun is technical, not success.
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
            # Only failure-checkpoint closure remains after cancelling the
            # alarm; the external wrapper's finite grace bounds this write.
            write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "replicate_index", "seed",
          "source_job_id", "source_baseline_validated", "rival_diagnostics_complete", "diagnostics",
          "elapsed_seconds", "error") if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
