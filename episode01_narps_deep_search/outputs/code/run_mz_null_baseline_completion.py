#!/usr/bin/env python3
"""Complete only null baseline35's unfinished reverse MZ fit after its timeout.

Read the preserved simulation and three converged original500 fits from the
selected failed worker. Validate and rescore exact copies, then fit reverse MZ
from the unchanged fitter's ordinary initialization. Never simulate, copy the
source payloads, overwrite the failed attempt, or admit repairs to old pools.
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

from hierarchical_behavior import fit_partially_pooled, load_split
from qualify_assets import require_compute
from run_single_process_recovery import HALVES, IMPLEMENTATION, DESIGN_SOURCE, validate_fit
import run_mz_null_replicate as indexed
import run_mz_null_rival as null
import summarize_mz_null_baseline_wave as baseline
import summarize_mz_fixed_menu as fixed


REPLICATE_INDEX = 35
SEED = indexed.null_seed(REPLICATE_INDEX)
SOURCE = baseline.source_directory(REPLICATE_INDEX, 46579924, 46580387)
SOURCE_REPORT = indexed.REPORT_NAME
SOURCE_JOB_ID = "46580478"
SOURCE_FINISHED_UTC = "2026-10-05T06:46:38.259015+00:00"
SOURCE_ERROR = "RuntimeError: Null representative interrupted by signal 15; unfinished work is technical"
REPORT_NAME = "mz_null_baseline_completion.json"
PHASE = "M0_null_MZ_pairwise_baseline_completion"
EXPECTED = tuple((model, train, held) for train, held in HALVES for model in ("M0", "MZ"))


def duration(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative recorded duration")
    return value


def completion_context(source, truth, people, spec):
    """Accept only this diagnosed timeout and its exact three-fit prefix."""
    if (spec.get("policy_version") != 6 or spec.get("state") != "recorded_before_model_scoring"
            or spec["optimizer"]["maximum_sweeps"] != 500
            or spec["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or spec["recovery_plan"]["primary_splits"] != [[1, 4], [2, 3]]
            or spec["contaminant"]["fraction"] != .02 or spec["contaminant"]["miss_probability"] != .01):
        raise ValueError("Preserve the original500 implementation and fixed200 null cell")
    baseline.validate_identity(source, REPLICATE_INDEX, spec)
    baseline.technical_metadata(source)
    if (str(source.get("job_id")) != SOURCE_JOB_ID or source.get("finished_utc") != SOURCE_FINISHED_UTC
            or source.get("internal_seconds") != 2700
            or source.get("status") != "technical_representative_failure" or source.get("error") != SOURCE_ERROR
            or source.get("scored_primary_directions") != [{"training_runs": [1, 4], "scoring_runs": [2, 3]}]
            or [source.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
        raise ValueError("Only worker46580478's diagnosed null35 signal15 three-fit timeout is selected")
    if not isinstance(truth, dict) or set(truth) != set(people):
        raise ValueError("Saved null truth and actual cohort participant identities differ")
    for params in truth.values():
        if (not isinstance(params, dict)
                or set(params) != {"kappa", "log_a", "nondecision_logit", "eta", "rho", "criterion"}
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in params.values())
                or any(params[key] != 0. for key in ("eta", "rho", "criterion"))):
            raise ValueError("Retain the finite centered M0 generating truth")
        for name in ("kappa", "log_a", "nondecision_logit"):
            low, high = spec["parameters"][name]["bounds"]
            if not low <= params[name] <= high:
                raise ValueError("Saved nuisance truth is outside the original support")
    if source.get("simulated_deadline_misses") != sum(int(row["no_response"]) for rows in people.values() for row in rows):
        raise ValueError("Saved deadline-miss accounting differs from the actual cohort")
    fixed.fit_panel(source, EXPECTED[:3], spec, people)
    previous = duration(source.get("elapsed_seconds"), "Previous process cost")
    costs = [duration(fit.get("elapsed_seconds"), "Preserved fit cost") for fit in source["fits"]]
    if sum(costs) > previous:
        raise ValueError("Preserved fit costs cannot exceed their containing previous process cost")
    return deepcopy(source["fits"]), {
        "source_context_validated": True,
        "source_finished_utc": source["finished_utc"], "source_started_utc": source["started_utc"],
        "source_status": source["status"], "source_error": source["error"],
        "previous_process_elapsed_seconds": previous,
        "original_simulation_seconds": duration(source.get("simulation_seconds"), "Original simulation cost"),
        "simulated_deadline_misses": source["simulated_deadline_misses"],
        "per_fit_provenance": [
            {"model": model, "training_runs": list(train), "scoring_runs": list(held),
             "mode": "validated_preserved_fit_rescored", "source_fit_index": position,
             "source_job_id": SOURCE_JOB_ID, "original_fit_elapsed_seconds": costs[position],
             "validation_maximum_sweeps": 500}
            for position, (model, train, held) in enumerate(EXPECTED[:3])
        ] + [{"model": "MZ", "training_runs": [2, 3], "scoring_runs": [1, 4],
              "mode": "fresh_fit_original_initialization", "validation_maximum_sweeps": 500}],
    }


def write_report(output, report):
    temporary = output / (REPORT_NAME + ".partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / REPORT_NAME)


def interrupted(signum, frame):
    raise RuntimeError(f"Null baseline completion interrupted by signal {signum}; unfinished work is technical")


def run(args):
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original behavioral implementation")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= 2700:
        raise ValueError("The completion whole-process cap is 1 through2700 seconds")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or any(os.environ.get(name) != "1" for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")):
        raise ValueError("Require one allocated CPU and all four numerical threads explicitly one")
    output = args.output_dir.resolve()
    protected = (SOURCE.resolve(), DESIGN_SOURCE.parent.resolve(), indexed.PILOT_DIRECTORY.resolve())
    if output.exists() or any(output == path or path in output.parents for path in protected):
        raise FileExistsError("Use a fresh output outside the original source, design and null pilot")
    output.mkdir(parents=True)
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Internal whole-process deadline exceeded")

    report = {
        "analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "M0", "fitted_models": ["M0", "MZ"],
        "replicate_index": REPLICATE_INDEX, "seed": SEED, "SESOI_level": 0.,
        "focal_generating_truth": dict(indexed.representative.NULL_TRUTH),
        "source_dir": str(SOURCE), "source_report": SOURCE_REPORT, "source_job_id": SOURCE_JOB_ID,
        "generating_parameters": str(SOURCE / "generating_parameters.json"),
        "simulated_events": str(SOURCE / "simulated_events.csv.gz"),
        "job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": datetime.now(timezone.utc).isoformat(),
        "internal_seconds": args.max_seconds, "status": "running", "pairwise_diagnostic_complete": False,
        "source_context_validated": False, "preserved_scores_recomputed": False,
        "saved_score_agreement": "not_evaluated_original_failure_has_no_saved_scores",
        "simulation_regenerated": False, "simulation_copied": False, "saved_completed_fits_refitted": False,
        "unfinished_fit_state_reused": False, "observed_choices_or_RTs_used": False,
        "old_synthetic_responses_used": False, "selected_replicates": [REPLICATE_INDEX],
        "wave_selected": False, "retry_wave_selected": False, "full_menu_wave_selected": False,
        "fixed_cohort_count": indexed.FIXED_COHORTS, "component_scope": "single_M0_MZ_baseline_completion",
        **{field: "not_evaluated" for field in null.CLAIM_FIELDS},
        "full_recovery_pass": None, "episode_terminal": False,
        "interpretation": "Only null35's missing reverse MZ fit is computed; the three preserved original500 fits are validated and rescored. One nested pairwise diagnostic is not full-menu selection, a rate, an interval or an early-kill verdict. Existing pool/rival source acceptance remains unchanged.",
        "cost_accounting": "Cumulative process cost is previous process elapsed plus this process elapsed once. Original fit costs are already included in the previous process; preserved fit elapsed fields retain their original meaning.",
        "process_state_boundary": "The coordinator establishes the original worker is terminal before selecting completion; report fields are not scheduler evidence.",
        "fits": [], "scored_primary_directions": [], "scored_fit_prefix": [],
        "preserved_fit_rescore_seconds": 0.,
    }
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded = False
    fresh_started = None
    try:
        write_report(output, report)
        spec = json.loads(args.implementation.read_text())
        source = json.loads((SOURCE / SOURCE_REPORT).read_text())
        truth = json.loads((SOURCE / "generating_parameters.json").read_text())
        events = SOURCE / "simulated_events.csv.gz"
        people = null.shared.load_cohort(events, check)
        check()
        context_started = time.perf_counter()
        preserved, metadata = completion_context(source, truth, people, spec)
        report.update(metadata, implementation=spec, fits=preserved,
                      source_validation_and_copy_seconds=time.perf_counter() - context_started,
                      participants=108, runs=432, trials=27648)
        # Copy time is included in the measured validation/copy stage; no old
        # elapsed field is overwritten or counted as newly executed work.
        write_report(output, report)
        scores = {person: {"M0": 0., "MZ": 0., "version": rows[0]["task_version"]}
                  for person, rows in people.items()}
        position = 0
        for train, held in HALVES:
            check()
            training, scoring = load_split(events, list(train), list(held))
            for model in ("M0", "MZ"):
                check()
                if position < 3:
                    fit = preserved[position]
                else:
                    fresh_started = time.perf_counter()
                    fit = fit_partially_pooled(training, model, "fixed", spec)
                    fit.update(training_runs=list(train), scoring_runs=list(held),
                               elapsed_seconds=time.perf_counter() - fresh_started)
                    report["fresh_fit_elapsed_seconds"] = fit["elapsed_seconds"]
                    fresh_started = None
                    json.dumps(fit, allow_nan=False)
                    report["fits"].append(fit)
                    write_report(output, report)
                    check()
                    validate_fit(fit, model, train, held, spec, people)
                score_started = time.perf_counter()
                for person, value in null.shared.score_fit(fit, scoring, spec, check).items():
                    scores[person][model] += value
                score_seconds = time.perf_counter() - score_started
                if position < 3:
                    report["preserved_fit_rescore_seconds"] += score_seconds
                else:
                    report["fresh_fit_score_seconds"] = score_seconds
                report["scored_fit_prefix"].append({"model": model, "training_runs": list(train), "scoring_runs": list(held)})
                position += 1
                report["preserved_scores_recomputed"] = position >= 3
                check()
                write_report(output, report)
            report["scored_primary_directions"].append({"training_runs": list(train), "scoring_runs": list(held)})
            write_report(output, report)
        fixed.fit_panel(report, EXPECTED, spec, people)
        fixed.score_panel({"participant_scores": scores}, ("M0", "MZ"))
        diagnostic = fixed.nested_contrast(scores, "MZ", "M0")
        check()
        report.update(participant_scores=scores, nested_comparison=diagnostic)
        write_report(output, report)
        check()
        succeeded = True
    except Exception as error:
        report.update(status="technical_baseline_completion_failure", error=f"{type(error).__name__}: {error}")
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if succeeded and time.monotonic() >= deadline:
            succeeded = False
            report.update(status="technical_baseline_completion_failure", error="RuntimeError: Internal deadline reached during finalization")
        if not succeeded:
            report.pop("participant_scores", None)
            report.pop("nested_comparison", None)
        if fresh_started is not None:
            report["unfinished_fresh_fit_elapsed_seconds"] = time.perf_counter() - fresh_started
        elapsed = time.perf_counter() - started
        report.update(status="pairwise_diagnostic_completed" if succeeded else report["status"],
                      pairwise_diagnostic_complete=succeeded, elapsed_seconds=elapsed,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if report["source_context_validated"]:
            report["cumulative_process_elapsed_seconds"] = report["previous_process_elapsed_seconds"] + elapsed
        write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=2700)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "replicate_index", "seed",
          "pairwise_diagnostic_complete", "nested_comparison", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error")
          if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["pairwise_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
