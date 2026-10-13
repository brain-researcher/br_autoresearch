#!/usr/bin/env python3
"""One descriptive twice-SESOI index0 menu, conditional on terminal MVC repair.

Read four explicit JSON inputs only. Existing baseline and failed-pilot
validators supply the original500 evidence; the new MVC pair is validated
under its private1500 specification without relabeling either report.
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
from types import SimpleNamespace

from qualify_assets import require_compute


EPISODE = Path(__file__).resolve().parents[2]
RUNS = EPISODE / "outputs/runs/behavioral_mechanism_kill_test"
IMPLEMENTATION = Path(__file__).with_name("behavioral_implementation.json")
BASE_REPORT = RUNS / "20261005-MZ-effect-level-representatives-46603942/twice/replicate0/mz_effect_level_recovery.json"
FAILED_REPORT = RUNS / "20261005-MZ-twice-menu-pilot-46634879-attempt001/twice_mz_menu_pilot.json"
REPAIR_JOB = "46642546"
REPAIR_REPORT = RUNS / f"20261005-MZ-twice-MVC-repair-{REPAIR_JOB}-attempt001/pilot/twice_mz_mvc_repair_pilot.json"
PHASE = "MZ_twice_SESOI_index0_repaired_menu_nested_diagnostics"
MAX_SECONDS = 240
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
DIAGNOSTICS = ("nested_comparisons", "nested_comparison_eligibility", "eligible_models_diagnostic",
               "maximal_eligible_models_diagnostic", "multiple_incomparable_eligible_models")


def load_helpers():
    import run_twice_mz_mvc_repair_pilot as repair
    import summarize_twice_mz_menu_wave as collector
    return SimpleNamespace(repair=repair, collector=collector, fixed=collector.fixed, twice=collector.twice)


def validate_repair(report, base, context, spec, helpers):
    repair, collector, fixed = helpers.repair, helpers.collector, helpers.fixed
    evidence = collector.terminal(report)
    technical = report.get("status") == "technical_mvc_repair_failure"
    if (evidence["job_id"] != REPAIR_JOB or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != repair.PHASE or report.get("generating_model") != "MZ"
            or report.get("fitted_model") != "MVC" or report.get("effect_level") != "twice"
            or report.get("SESOI_level") != 2. or report.get("generating_z") != .40
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != 0
            or type(report.get("seed")) is not int or report["seed"] != 1900057
            or report.get("selected_replicates") != [0] or report.get("source_report") != str(BASE_REPORT)
            or report.get("source_dir") != str(BASE_REPORT.parent) or report.get("failed_report") != str(FAILED_REPORT)):
        raise ValueError("Only the explicitly selected twice0 MVC repair and its exact sources are admitted")
    if (report.get("workers") != 4 or report.get("allocated_cpus") != 4
            or report.get("thread_environment") != {name: "1" for name in THREAD_ENV}
            or report.get("original_maximum_sweeps") != 500 or report.get("runtime_maximum_sweeps") != 1500
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= 5400
            or any(report.get(field) != "not_evaluated" for field in repair.CLAIMS)
            or report.get("full_recovery_pass") is not None
            or any(report.get(field) is not False for field in ("episode_terminal", "simulation_regenerated",
                "saved_M0_MZ_refitted", "saved_fit_parameters_reused", "observed_choices_or_RTs_used",
                "source_files_modified", "full_menu_wave_selected", "retry_wave_selected", "automatic_retry", "automatic_resume"))
            or any(field in report for field in DIAGNOSTICS)):
        raise ValueError("Retain four-worker execution, the declared numerical ceiling and closed component claims")
    if technical:
        if report.get("mvc_repair_complete") is not False or not isinstance(report.get("error"), str) or not report["error"]:
            raise ValueError("Technical repair requires an explicit error and no complete-process claim")
        evidence["error"] = report["error"]
    elif report.get("status") != "mvc_repair_completed" or report.get("mvc_repair_complete") is not True:
        raise ValueError("The coordinator must select an actually terminal completed or technical repair")
    runtime = repair.runtime_specification(spec)
    if ((report.get("implementation") is not None and report["implementation"] != spec)
            or (report.get("runtime_implementation") is not None and report["runtime_implementation"] != runtime)):
        raise ValueError("Only a private500-to1500 maximum-sweeps change is permitted")
    fits, scored, completed = (report.get(field) for field in ("fits", "scored_components", "completed_rivals"))
    if (not isinstance(fits, list) or len(fits) > 2 or not isinstance(scored, list)
            or not 0 <= len(fits) - len(scored) <= 1 or completed not in ([], ["MVC"])
            or (completed == ["MVC"] and len(scored) != 2)
            or type(report.get("fresh_fit_calls")) is not int
            or not len(fits) <= report["fresh_fit_calls"] <= min(2, len(fits) + 1)):
        raise ValueError("Retain the ordered two-fit/scored prefix and fresh-call count")
    halves = helpers.twice.core.HALVES
    if report.get("scored_primary_directions") != [
            {"training_runs": list(train), "scoring_runs": list(held)} for train, held in halves[:len(scored)]]:
        raise ValueError("Scored direction identity differs from the actual prefix")
    evidence.update(fresh_fit_calls=report["fresh_fit_calls"], completed_rivals=deepcopy(completed),
        returned_fits=[{**{key: deepcopy(fit.get(key)) for key in ("model", "training_runs", "scoring_runs",
            "converged", "elapsed_seconds", "support_contacts", "subject_optimizer_failures")},
            "sweeps": len(fit["objective_history"]) if isinstance(fit.get("objective_history"), list) else None} for fit in fits],
        scored_components=[{key: deepcopy(value) for key, value in row.items() if key != "participant_contributions"} for row in scored],
        parallel_failure=deepcopy(report.get("parallel_failure")),
        parallel_workers_cleaned_up=report.get("parallel_workers_cleaned_up"))
    validated, contextual, replayed = (report.get(field) for field in
        ("source_baseline_validated", "repair_context_validated", "saved_scores_replayed"))
    if any(type(value) is not bool for value in (validated, contextual, replayed)):
        raise ValueError("Source, context and replay states must be explicit")
    if not validated:
        if not technical or contextual or replayed or fits or scored or completed or report["fresh_fit_calls"] or "participant_scores" in report:
            raise ValueError("Unvalidated source cannot supply fitted or scored evidence")
        return None, evidence
    if (report.get("implementation") != spec or report.get("runtime_implementation") != runtime
            or report.get("source_job_id") != str(base["job_id"])
            or report.get("source_finished_utc") != base["finished_utc"]
            or report.get("focal_generating_truth") != base["focal_generating_truth"]
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]
            or report.get("source_process_elapsed_seconds") != base["elapsed_seconds"]
            or report.get("source_fit_elapsed_seconds") != [fit["elapsed_seconds"] for fit in base["fits"]]):
        raise ValueError("Validated baseline identity, original500 specification or cost references changed")
    if not contextual:
        if not technical or replayed or fits or scored or completed or report["fresh_fit_calls"] or "participant_scores" in report:
            raise ValueError("Failed-context admission must precede replay and fitting")
        return None, evidence
    if report.get("numerical_repair") != context:
        raise ValueError("Exact original failure, preserved model prefix or numerical repair context changed")
    if not replayed:
        if not technical or fits or scored or completed or report["fresh_fit_calls"] or "participant_scores" in report:
            raise ValueError("No new fitted or scored evidence is admitted before original-score replay")
        return None, evidence
    if (report.get("saved_score_replay_absolute_tolerance") != 1e-12
            or collector.duration(report.get("saved_score_replay_max_absolute_discrepancy")) > 1e-12):
        raise ValueError("Require the original successful1e-12 saved-score replay")
    baseline = fixed.score_panel(base, ("M0", "MZ"))
    panel = fixed.score_panel(report, ("M0", "MZ", *completed))
    if set(panel) != set(baseline) or any(panel[p][key] != baseline[p][key]
            for p in baseline for key in ("version", "M0", "MZ")):
        raise ValueError("Original saved score last bits or participant/version roster changed")
    for number, (fit, (train, held)) in enumerate(zip(fits, halves)):
        if (fit.get("model") != "MVC" or fit.get("rival") != "fixed" or fit.get("participants") != sorted(baseline)
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
                or type(fit.get("converged")) is not bool):
            raise ValueError("Returned fit must retain its actual MVC/primary-direction identity")
        collector.duration(fit.get("elapsed_seconds"))
        if number < len(scored):
            fixed.validate_fit(fit, "MVC", train, held, runtime, baseline)
    for direction, (row, (train, held)) in enumerate(zip(scored, halves)):
        if (row.get("model") != "MVC" or row.get("direction") != direction
                or row.get("training_runs") != list(train) or row.get("scoring_runs") != list(held)
                or row.get("heldout_trials_per_participant") != 128 or row.get("per_participant_denominator") != 256
                or not isinstance(row.get("participant_contributions"), dict)
                or set(row["participant_contributions"]) != set(baseline)
                or not all(helpers.twice.finite(value) for value in row["participant_contributions"].values())):
            raise ValueError("Each scored half requires the unchanged108-person128/256 contribution")
    if completed and any(panel[p]["MVC"] != scored[0]["participant_contributions"][p]
            + scored[1]["participant_contributions"][p] for p in baseline):
        raise ValueError("The completed MVC scores must equal both retained contributions")
    if technical:
        return None, evidence
    if (len(fits) != 2 or len(scored) != 2 or completed != ["MVC"] or report["fresh_fit_calls"] != 2
            or report.get("parallel_workers_cleaned_up") is not True or report.get("parallel_failure") is not None
            or report.get("active_fit") is not None
            or report.get("total_fresh_fit_seconds") != sum(fit["elapsed_seconds"] for fit in fits)):
        raise ValueError("Complete admission requires the entire fresh pair and successful worker cleanup")
    return {p: panel[p]["MVC"] for p in panel}, evidence


def empty_summary(spec=None):
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "implementation": spec,
        "status": "unknown", "reason": "input_not_yet_validated", "menu_complete": False,
        "generating_model": "MZ", "effect_level": "twice", "SESOI_level": 2., "generating_z": .40,
        "replicate_index": 0, "seed": 1900057, "selected_replicates": [0], "fixed_cell_count": 200,
        "input_reports": {"baseline": str(BASE_REPORT), "original_pilot": str(FAILED_REPORT), "MVC_repair": str(REPAIR_REPORT)},
        **{key: "not_evaluated" for key in ("family_selection", "family_recovery_rate", "confusion_rate", "group_recovery",
            "interval_coverage", "full_gate", "full_menu_selection")}, "full_recovery_pass": None, "episode_terminal": False,
        "new_fits": False, "new_simulation": False, "source_events_read": False, "saved_scores_rescored": False,
        "wave_selected": False, "automatic_retry": False,
        "interpretation": "Index0 descriptive nine-edge menu only. Original500 M0/MZ and MC/MV/MVZ remain separate from the private1500 MVC repair. No family, rate, interval or full-gate verdict; no admission from a technical repair, even when both halves survived.",
        "cost_accounting": "Historical process and included fit durations are non-additive references; no allocation is charged by this summary."}


def build_summary(base, failed, report, spec, helpers, check=lambda: None):
    result = empty_summary(spec)
    helpers.twice.validate_identity(base, 0, spec)
    helpers.twice.completed_diagnostic(base, spec)
    baseline = helpers.fixed.score_panel(base, ("M0", "MZ"))
    check()
    context = helpers.repair.repair_context(failed, base, baseline, spec, helpers)
    result["cost_references"] = {"baseline_process_seconds": base["elapsed_seconds"],
        "original_pilot_process_seconds": context["previous_process_elapsed_seconds"],
        "original_pilot_fit_seconds_included_in_process": context["previous_fit_elapsed_seconds"]}
    result["original_pilot_evidence"] = context["previous_terminal_evidence"]
    check()
    mvc, evidence = validate_repair(report, base, context, spec, helpers)
    result["repair_evidence"] = evidence
    result["cost_references"]["MVC_repair_process_seconds"] = evidence["elapsed_seconds"]
    if mvc is None:
        result["reason"] = "technical_MVC_repair_unknown"
        return result
    scores = deepcopy(context["previous_participant_scores"])
    for person in scores:
        scores[person]["MVC"] = mvc[person]
    check()
    result.update(status="complete_menu", menu_complete=True, participants=108, runs=432, trials=27648,
        model_menu=list(helpers.fixed.MODELS), participant_scores=scores,
        original_maximum_sweeps=500, MVC_runtime_maximum_sweeps=1500,
        **helpers.repair.pilot.menu_diagnostics(scores, SimpleNamespace(menu=helpers.fixed)))
    result.pop("reason")
    return result


def write_report(output, result):
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if (args.implementation.resolve() != IMPLEMENTATION.resolve() or args.repair_report.resolve() != REPAIR_REPORT.resolve()
            or not 1 <= args.max_seconds <= MAX_SECONDS or any(os.environ.get(key) != "1" for key in THREAD_ENV)):
        raise ValueError("Require the exact selected sources, six actual thread limits one and a1..240-second cap")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        raise FileExistsError("Choose a fresh summary; historical reports are immutable")
    started, deadline = time.perf_counter(), time.monotonic() + args.max_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = empty_summary()

    def check():
        if time.monotonic() >= deadline:
            raise RuntimeError("Whole-process repaired-menu summary deadline exceeded")

    def interrupted(signum, frame):
        raise RuntimeError(f"Repaired-menu summary interrupted by signal {signum}")

    def publish():
        result.update(recorded_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started,
                      internal_seconds=args.max_seconds, job_id=os.environ.get("SLURM_JOB_ID"))
        write_report(args.output, result)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    signal.alarm(args.max_seconds)
    succeeded, failure = False, None
    try:
        check()
        helpers = load_helpers()
        spec = json.loads(IMPLEMENTATION.read_text())
        result["implementation"] = spec
        reports = []
        for path in (BASE_REPORT, FAILED_REPORT, REPAIR_REPORT):
            check()
            reports.append(json.loads(path.read_text()))
            check()
        result = build_summary(*reports, spec, helpers, check)
        publish()
        check()
        succeeded = True
    except FileNotFoundError as error:
        result.update(reason="selected_terminal_report_missing", missing_report=str(error.filename or error))
        publish()
        # Missing evidence is an unknown menu, not a scientific failure.
        try:
            check()
            succeeded = True
        except BaseException as error:
            failure = f"{type(error).__name__}: {error}"
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        signal.alarm(0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if succeeded:
            try:
                check()
            except BaseException as error:
                succeeded, failure = False, f"{type(error).__name__}: {error}"
        if not succeeded:
            for field in (*DIAGNOSTICS, "participant_scores", "model_menu"):
                result.pop(field, None)
            result.update(status="unknown", reason="technical_summary_failure", menu_complete=False, error=failure)
            publish()
    print(json.dumps({key: result[key] for key in ("status", "menu_complete", "reason") if key in result}), flush=True)
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
