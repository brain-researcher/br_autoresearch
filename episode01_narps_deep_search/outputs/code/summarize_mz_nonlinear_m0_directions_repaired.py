#!/usr/bin/env python3
"""Pool two explicit nonlinear M0 directions with the preserved repaired MZ.

M0 directions contribute their original128-trial log sums divided by256;
both are required before addition. Numerical ceilings and historical process
durations remain distinct from scientific contrasts and allocation accounting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_nonlinear_m0_direction_repair as repair
import summarize_mz_nonlinear_both_repaired as previous


nonlinear, original = previous.nonlinear, previous.original
SOURCE_REPORT = previous.m0_pool.SOURCE_REPORT
ORIGINAL_M0_REPORT = previous.m0_pool.repair.FAILED_REPORT
FAILED_M0_REPAIR_REPORT = previous.M0_REPORT
MZ_JOB = "46612046"
MZ_REPORT = previous.reverse.SOURCE.parent / f"20261005-MZ-nonlinear-MZ-reverse-repair-{MZ_JOB}" / previous.reverse.REPORT_NAME
PHASE = "MZ0_nonlinear_M0_independent_directions_and_preserved_MZ_nested_diagnostic"
DIRECTION_NORMALIZATION = "one128_trial_heldout_log_likelihood_sum_divided_by256_per_participant"


def report_paths(repair_dir, job_id):
    job = original.parent_job(job_id)
    root = Path(repair_dir).resolve()
    expected = nonlinear.SOURCE.parent / f"20261005-MZ-nonlinear-M0-directions-repair-{job}"
    if root != expected.resolve():
        raise ValueError("Use the explicit selected nonlinear M0 direction-repair parent")
    return {direction: root / f"dir{direction}" / repair.REPORT_NAME for direction in (0, 1)}


def validate_direction(report, direction, job_id, spec, primary, source, context, original_context):
    train, held = nonlinear.HALVES[direction]
    runtime = repair.runtime_specification(spec, direction)
    technical = report.get("status") == "technical_nonlinear_failure"
    expected = {"analysis_id": "behavioral_mechanism_kill_test", "phase": repair.PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": nonlinear.shared.SEED,
        "direction": direction, "fitted_model": "M0", "rival": "nonlinear",
        "source_dir": str(nonlinear.SOURCE), "source_report": str(SOURCE_REPORT),
        "failed_report": str(FAILED_M0_REPAIR_REPORT),
        "training_runs": list(train), "scoring_runs": list(held),
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": runtime["optimizer"]["maximum_sweeps"],
        "numerical_override": repair.numerical_override(direction),
        "score_normalization": DIRECTION_NORMALIZATION, "direction_contribution_only": True,
        "heldout_trials_per_direction_per_participant": 128, "combined_scored_trials_per_participant": 256,
        "workers": 4, "allocated_cpus": 4, "parallel_execution": True, "selected_replicates": [0]}
    if (any(report.get(key) != value for key, value in expected.items())
            or str(report.get("job_id")) != str(job_id)
            or type(report.get("direction")) is not int or type(report.get("replicate_index")) is not int
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= repair.MAX_SECONDS
            or any(report.get(field) != "not_evaluated" for field in nonlinear.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_nonlinear", "observed_choices_or_RTs_used", "wave_selected",
                "retry_wave_selected", "automatic_retry", "episode_terminal"))
            or "cumulative_process_elapsed_seconds" in report):
        raise ValueError("Preserve the selected single direction, actual runtime and closed claims")
    if "implementation" in report:
        if (report["implementation"] != spec or report.get("original_implementation") != spec
                or report.get("runtime_implementation") != runtime):
            raise ValueError("Retain original500 separately from the direction-specific optimizer ceiling")
    elif (not technical or report.get("original_implementation") is not None
            or report.get("runtime_implementation") is not None):
        raise ValueError("Only early technical checkpoints may omit loaded implementations")
    elapsed = original.duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid terminal times")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Only explicit terminal directions may be pooled") from error
    contextual, validated, replayed = (report.get(key) for key in
        ("repair_context_validated", "source_baseline_validated", "saved_primary_scores_replayed"))
    if any(type(value) is not bool for value in (contextual, validated, replayed)):
        raise ValueError("Keep context, source and replay qualification explicit")
    if contextual:
        if (not validated or report.get("prior_1500_failed_context") != context
                or report.get("original_500_failed_context") != original_context):
            raise ValueError("Reconstruct both exact original500 and failed1500 M0 lineages")
    elif (report.get("prior_1500_failed_context") is not None
            or report.get("original_500_failed_context") is not None):
        raise ValueError("Unvalidated contexts cannot supply prior evidence")
    if validated:
        if (report.get("source_job_id") != source["job_id"]
                or report.get("source_finished_utc") != source["finished_utc"]
                or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
            raise ValueError("Retain the exact original complete108-person source")
    elif not technical or replayed:
        raise ValueError("Completed direction scores require a validated source")
    if replayed:
        discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_primary_participant_scores") != primary
                or report.get("saved_score_replay_absolute_tolerance") != nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE
                or type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
                or not 0 <= discrepancy <= nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("Preserve original primary scores and replay")
    elif not technical or "saved_primary_participant_scores" in report:
        raise ValueError("New nonlinear direction fits require primary replay")
    fits, scored, completed, calls = (report.get(key) for key in
        ("fits", "scored_primary_directions", "completed_direction_scores", "fresh_fit_calls"))
    expected_scored = [{"training_runs": list(train), "scoring_runs": list(held)}]
    if (not isinstance(fits, list) or len(fits) > 1 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored)+1
            or scored != expected_scored[:len(scored)] or not isinstance(completed, dict)
            or bool(completed) != bool(scored) or type(calls) is not int or calls not in (0, 1)
            or not len(fits) <= calls <= len(fits)+1
            or ((calls or fits or completed) and not (contextual and validated and replayed))):
        raise ValueError("Retain one fresh fit and its actual selected-direction score prefix")
    returned = []
    for fit in fits:
        if (fit.get("model") != "M0" or fit.get("rival") != "nonlinear"
                or fit.get("participants") != sorted(primary) or fit.get("training_runs") != list(train)
                or fit.get("scoring_runs") != list(held)):
            raise ValueError("Keep the selected five-parameter nonlinear M0 direction")
        consumed = original.duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            nonlinear.validate_nonlinear_fit(fit, "M0", train, held, runtime, primary)
        elif fit.get("converged") is not False or scored or not technical:
            raise ValueError("Only a converged direction may supply scores")
        returned.append({"converged": fit["converged"], "elapsed_seconds": consumed,
                         "support_contacts": fit.get("support_contacts")})
    if completed:
        panel = nonlinear.fixed.score_panel({"participant_scores": completed}, (nonlinear.SCORE_KEY,))
        if set(panel) != set(primary) or any(panel[p]["version"] != primary[p]["version"] for p in primary):
            raise ValueError("All108 original participants must supply finite direction contributions")
    row = {"direction": direction, "fitted_model": "M0", "job_id": str(job_id), "phase": repair.PHASE,
        "elapsed_seconds": elapsed, "finished_utc": report["finished_utc"], "fresh_fit_calls": calls,
        "fitted_maximum_sweeps": runtime["optimizer"]["maximum_sweeps"], "workers": 4,
        "returned_fits": returned, "scored_primary_directions": scored}
    if technical:
        if (report.get("nonlinear_direction_complete") is not False or "participant_scores" in report
                or not isinstance(report.get("error"), str) or not report["error"]):
            raise ValueError("Technical directions remain unknown without final scores")
        row.update(status="technical_unknown", error=report["error"], completed_direction_scores=completed)
        return row, None
    if (report.get("status") != "nonlinear_direction_completed" or report.get("nonlinear_direction_complete") is not True
            or not (contextual and validated and replayed) or calls != 1 or len(fits) != 1 or scored != expected_scored
            or report.get("participant_scores") != completed or report.get("parallel_workers_cleaned_up") is not True):
        raise ValueError("Completed directions require one qualified fit, complete contribution panel and finished workers")
    row["status"] = "complete_component"
    return row, completed


def pool(repair_dir, job_id, spec):
    paths = report_paths(repair_dir, job_id)
    if spec["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Keep the original500 source specification")
    source = json.loads(SOURCE_REPORT.read_text())
    primary = nonlinear.fixed.validate_base(source, spec)
    original_m0 = json.loads(ORIGINAL_M0_REPORT.read_text())
    original_context = previous.m0_pool.repair.repair_context(original_m0, source, primary, spec)
    failed_m0 = json.loads(FAILED_M0_REPAIR_REPORT.read_text())
    context = repair.repair_context(failed_m0, source, primary, spec)
    original_mz = json.loads(previous.reverse.FAILED_REPORT.read_text())
    first, mz_context = previous.reverse.repair_context(original_mz, source, primary, spec)
    mz_report = json.loads(MZ_REPORT.read_text())
    mz_row, mz_scores, saved, identity = previous.validate_reverse(mz_report, MZ_JOB, spec, first, mz_context)
    if (mz_scores is None or saved != primary or identity != (str(source["job_id"]), source["finished_utc"])):
        raise ValueError("The preserved46612046 MZ component must be complete on the same original source")
    mz_row["report"] = str(MZ_REPORT)
    rows, panels = [], {}
    for direction, path in paths.items():
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            rows.append({"direction": direction, "fitted_model": "M0", "status": "missing_unknown", "report": str(path)})
            continue
        row, scores = validate_direction(report, direction, job_id, spec, primary, source, context, original_context)
        row["report"] = str(path)
        rows.append(row)
        if scores is not None:
            panels[direction] = scores
    complete = len(panels) == 2
    historical = [
        {"role": role, "report": str(path), "job_id": str(report["job_id"]),
         "elapsed_seconds": original.duration(report["elapsed_seconds"])}
        for role, path, report in (("original_M0_500_failure", ORIGINAL_M0_REPORT, original_m0),
            ("M0_1500_failure", FAILED_M0_REPAIR_REPORT, failed_m0),
            ("original_MZ_reverse500_failure", previous.reverse.FAILED_REPORT, original_mz),
            ("completed_MZ_reverse1500_repair", MZ_REPORT, mz_report))]
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": nonlinear.shared.SEED, "rival": "nonlinear",
        "repair_dir": str(Path(repair_dir).resolve()), "repair_job_id": str(job_id), "MZ_repair_job_id": MZ_JOB,
        "source_dir": str(nonlinear.SOURCE), "source_report": str(SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_implementation": spec,
        "M0_runtime_implementations": {str(d): repair.runtime_specification(spec, d) for d in (0, 1)},
        "per_model_half_maximum_sweeps": {"M0": [2000, 1500], "MZ": [500, 1500]},
        "prior_failures": {"M0_original500": original_context, "M0_1500": context, "MZ_original500": mz_context},
        "historical_processes": historical,
        "historical_process_elapsed_seconds": sum(row["elapsed_seconds"] for row in historical),
        "new_M0_direction_process_elapsed_seconds": sum(row.get("elapsed_seconds", 0.) for row in rows),
        "cost_accounting": "Each of the four listed historical processes is counted once by its own elapsed_seconds, never by cumulative context fields. Each new M0 direction process appears once. Fit/copy/rescore/worker times are not added again; these descriptive wall durations do not replace coordinator allocation accounting.",
        "status": "nonlinear_pair_diagnostic_completed" if complete else "incomplete_nonlinear_M0_directions",
        "pooled_diagnostic_complete": complete, "complete_M0_directions": sorted(panels),
        "unknown_M0_directions": [d for d in (0, 1) if d not in panels],
        "components": rows + [mz_row], "fixed_primary_participant_scores": primary,
        "selected_replicates": [0], "wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in nonlinear.CLAIM_FIELDS},
        "fixed_vs_nonlinear_arbitration": "not_evaluated",
        "interpretation": "One saved-cohort nonlinear MZ-minus-M0 diagnostic. Fresh M0 directions use2000/1500 sweep ceilings; preserved completed MZ uses original-first500/reverse1500. Sum both M0 direction contributions, each already normalized by256, before comparison with full MZ scores. Unequal powers make M0 a curvature-adjusted reference, not globally symmetric valuation. Original and new scale contacts remain visible. No nonnested winner, family, recovery rate, interval, curtailment, robustness or gate verdict."}
    if complete:
        scores = {person: {"version": primary[person]["version"],
            "M0": sum(panels[d][person][nonlinear.SCORE_KEY] for d in (0, 1)),
            "MZ": mz_scores[person][nonlinear.SCORE_KEY]} for person in sorted(primary)}
        nonlinear.fixed.score_panel({"participant_scores": scores}, original.MODELS)
        result.update(participants=108, combined_heldout_trials_per_participant=256,
            score_normalization=nonlinear.NORMALIZATION, nonlinear_participant_scores=scores,
            nonlinear_nested_diagnostic=nonlinear.fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--repair-job-id", type=original.parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != nonlinear.IMPLEMENTATION.resolve():
        raise ValueError("Use the original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.repair_dir, args.repair_job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("phase", "status", "pooled_diagnostic_complete", "unknown_M0_directions")}, indent=2))
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
