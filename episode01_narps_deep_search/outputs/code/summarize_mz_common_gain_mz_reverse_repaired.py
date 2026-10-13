#!/usr/bin/env python3
"""Pool three preserved components with the one fresh reverse-MZ repair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from qualify_assets import require_compute
import run_mz_common_gain_mz_reverse_repair as repair
import summarize_mz_common_gain_range_repaired as old


common, original, fixed = old.common, old.original, old.fixed
OLD_JOB = "46615238"
OLD_ROOT = common.SOURCE.parent / f"20261005-MZ-common-gain-repair-{OLD_JOB}"
SOURCE_REPORT = old.SOURCE_REPORT
PHASE = "MZ0_common_gain_reverse_MZ_repaired_nested_diagnostic"
MODELS = ("M0", "MZ")


def repair_path(repair_dir, job_id):
    job = original.parent_job(job_id)
    root = Path(repair_dir).resolve()
    expected = common.SOURCE.parent / f"20261005-MZ-common-gain-MZ-reverse-repair-{job}" / "dir1" / "MZ"
    if root != expected.resolve():
        raise ValueError("Use the explicit new reverse-MZ repair directory and parent")
    return root / repair.REPORT_NAME


def validate_repair(report, job_id, spec, primary, source, context, m0_context):
    train, held = common.selected_runs(1)
    runtime = repair.runtime_specification(spec)
    technical = report.get("status") == "technical_common_gain_failure"
    expected = {"analysis_id": "behavioral_mechanism_kill_test", "phase": repair.PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": common.shared.SEED,
        "direction": 1, "fitted_model": "MZ", "fitted_models": ["MZ"], "rival": "fixed",
        "source_dir": str(common.SOURCE), "source_report": str(SOURCE_REPORT),
        "failed_report": str(repair.FAILED_REPORT),
        "training_runs": list(train), "scoring_runs": list(held),
        "gain_range_dollars": [10, 20], "filter_boundary_inclusive": True,
        "score_normalization": common.NORMALIZATION,
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": 1500,
        "numerical_override": repair.numerical_override(), "workers": 4, "allocated_cpus": 4,
        "parallel_execution": True, "selected_replicates": [0]}
    if (any(report.get(key) != value for key, value in expected.items())
            or str(report.get("job_id")) != str(job_id)
            or type(report.get("direction")) is not int or type(report.get("replicate_index")) is not int
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= repair.MAX_SECONDS
            or any(report.get(field) != "not_evaluated" for field in common.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_common_gain", "observed_choices_or_RTs_used", "wave_selected",
                "retry_wave_selected", "automatic_retry", "task_versions_are_independent_replications", "episode_terminal"))
            or "cumulative_process_elapsed_seconds" in report):
        raise ValueError("Preserve the one selected parallel reverse-MZ repair and closed scientific claims")
    if "implementation" in report:
        if (report["implementation"] != spec or report.get("original_implementation") != spec
                or report.get("runtime_implementation") != runtime):
            raise ValueError("Preserve original500 validation/scoring and only1500 fitting")
    elif (not technical or report.get("original_implementation") is not None
            or report.get("runtime_implementation") is not None):
        raise ValueError("Only an early technical checkpoint may omit loaded specifications")
    elapsed = original.duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid terminal times")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Require an explicit terminal repair") from error
    contextual, validated, replayed = (report.get(key) for key in
        ("source_context_validated", "source_baseline_validated", "saved_primary_scores_replayed"))
    if any(type(value) is not bool for value in (contextual, validated, replayed)):
        raise ValueError("Source and replay qualification must be explicit")
    if contextual:
        if (not validated or report.get("prior_MZ_failed_context") != context
                or report.get("original_M0_failed_context") != m0_context):
            raise ValueError("Reconstruct exact distinct MZ and original M0 failure contexts")
    elif (report.get("prior_MZ_failed_context") is not None
            or report.get("original_M0_failed_context") is not None):
        raise ValueError("Unqualified failures cannot supply context evidence")
    if validated:
        if (report.get("source_job_id") != source["job_id"]
                or report.get("source_finished_utc") != source["finished_utc"]
                or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
            raise ValueError("Use the complete unchanged108-person source")
    elif not technical or replayed:
        raise ValueError("A completed repair requires source qualification")
    if replayed:
        discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_primary_participant_scores") != primary
                or report.get("saved_score_replay_absolute_tolerance") != common.shared.REPLAY_ABSOLUTE_TOLERANCE
                or not original.finite(discrepancy)
                or not 0 <= discrepancy <= common.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("Retain the original full-range primary replay")
    elif not technical or "saved_primary_participant_scores" in report:
        raise ValueError("New fitting requires validated primary replay")
    fits, scored, completed, calls = (report.get(key) for key in
        ("fits", "scored_fits", "completed_model_scores", "fresh_fit_calls"))
    expected_score = [{"model": "MZ", "training_runs": list(train), "scoring_runs": list(held)}]
    if (not isinstance(fits, list) or len(fits) > 1 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored)+1
            or scored != expected_score[:len(scored)] or not isinstance(completed, dict)
            or set(completed) != ({"MZ"} if scored else set())
            or type(calls) is not int or calls not in (0, 1) or not len(fits) <= calls <= len(fits)+1
            or ((calls or fits or completed) and not (contextual and validated and replayed))):
        raise ValueError("Retain exactly the one fresh fit and its genuine scored prefix")
    support = report.get("retained_support")
    if support is not None:
        if not replayed or support != context["retained_support"]:
            raise ValueError("The repaired component must retain exactly the failed source slots")
        original.validate_support(support, primary, train, held)
    elif not technical or calls or fits or completed:
        raise ValueError("Fitting requires complete unchanged retained support")
    for fit in fits:
        if (fit.get("model") != "MZ" or fit.get("rival") != "fixed" or fit.get("participants") != sorted(primary)
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)):
            raise ValueError("Preserve reverse-only training/scoring identities")
        original.duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            common.validate_fit(fit, "MZ", train, held, runtime, primary)
        elif fit.get("converged") is not False or scored or not technical:
            raise ValueError("A nonconverged fit cannot supply scored evidence")
    if completed:
        panel = completed["MZ"]
        if not isinstance(panel, dict) or set(panel) != set(primary):
            raise ValueError("The score prefix must retain all108 participants")
        for person, value in panel.items():
            if (set(value) != {"log_sum", "trials"} or not original.finite(value["log_sum"])
                    or type(value["trials"]) is not int or value["trials"] != support[person]["heldout_trials"]):
                raise ValueError("Preserve finite raw log sums and actual trial counts")
    row = {"direction": 1, "fitted_model": "MZ", "job_id": str(job_id), "phase": repair.PHASE,
        "elapsed_seconds": elapsed, "finished_utc": report["finished_utc"], "fresh_fit_calls": calls,
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": 1500,
        "workers": 4, "parallel_execution": True}
    if technical:
        if (report.get("common_gain_component_complete") is not False or "participant_scores" in report
                or not isinstance(report.get("error"), str) or not report["error"]):
            raise ValueError("Technical repair outcomes remain explicitly unknown")
        row.update(status="technical_unknown", error=report["error"], scored_fits=scored)
        return row, None
    if (report.get("status") != "common_gain_component_completed" or report.get("common_gain_component_complete") is not True
            or not (contextual and validated and replayed) or calls != 1 or len(fits) != 1 or scored != expected_score):
        raise ValueError("Completed repair requires the full fresh fit and score panel")
    scores = report.get("participant_scores")
    if not isinstance(scores, dict) or set(scores) != set(primary):
        raise ValueError("The final reverse MZ panel must contain all108 participants")
    for person, score in scores.items():
        if (set(score) != {"version", "heldout_trials", "MZ_log_sum"}
                or score["version"] != primary[person]["version"]
                or type(score["heldout_trials"]) is not int or score["heldout_trials"] != support[person]["heldout_trials"]
                or not original.finite(score["MZ_log_sum"])
                or score["MZ_log_sum"] != completed["MZ"][person]["log_sum"]):
            raise ValueError("Final scores must equal the actual retained raw sums/counts")
    row["status"] = "complete_component"
    return row, scores


def pool(repair_dir, job_id, spec):
    new_path = repair_path(repair_dir, job_id)
    if spec["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Read the unchanged original implementation")
    source = json.loads(SOURCE_REPORT.read_text())
    primary = fixed.validate_base(source, spec)
    contexts = {direction: old.repair.repair_context(
        json.loads(old.repair.failed_path(direction).read_text()), source, primary, spec, direction)
        for direction in (0, 1)}
    for person in primary:
        a, b = (contexts[d]["retained_support"][person] for d in (0, 1))
        if ({tuple(slot) for slot in a["training_slots"]} != {tuple(slot) for slot in b["heldout_slots"]}
                or {tuple(slot) for slot in a["heldout_slots"]} != {tuple(slot) for slot in b["training_slots"]}):
            raise ValueError("Require complementary original filtered trial panels")
    historical, panels, component_rows = [], {}, []
    reports = {}
    for key, path in old.report_paths(OLD_ROOT, OLD_JOB).items():
        direction, model = key
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            if key == (1, "MZ"):
                raise ValueError("The exact failed reverse MZ is required to qualify a repair")
            row = {"direction": direction, "fitted_model": model, "status": "missing_unknown", "report": str(path)}
            historical.append(row)
            component_rows.append(row)
            continue
        reports[key] = report
        row, scores, _ = old.validate_component(report, direction, model, OLD_JOB, spec, primary, source, contexts[direction])
        row["report"] = str(path)
        historical.append(row)
        if key != (1, "MZ"):
            component_rows.append(row)
            if scores is not None:
                panels[key] = scores
    context = repair.repair_context(reports[1, "MZ"], source, primary, spec)
    try:
        report = json.loads(new_path.read_text())
    except FileNotFoundError:
        new_row, scores = {"direction": 1, "fitted_model": "MZ", "status": "missing_unknown"}, None
    else:
        new_row, scores = validate_repair(report, job_id, spec, primary, source, context, contexts[1])
    new_row["report"] = str(new_path)
    component_rows.append(new_row)
    if scores is not None:
        panels[1, "MZ"] = scores
    keys = [(d, m) for d in (0, 1) for m in MODELS]
    complete = len(panels) == 4
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": common.shared.SEED,
        "job_id": str(job_id), "repair_dir": str(Path(repair_dir).resolve()),
        "historical_job_id": OLD_JOB, "historical_root": str(OLD_ROOT),
        "source_dir": str(common.SOURCE), "source_report": str(SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_implementation": spec, "new_repair_runtime_implementation": repair.runtime_specification(spec),
        "per_model_direction_maximum_sweeps": {"M0": [1500, 1500], "MZ": [500, 1500]},
        "gain_range_dollars": [10, 20], "filter_boundary_inclusive": True,
        "prior_direction_failures": contexts, "prior_MZ_failed_context": context,
        "unique_prior_process_elapsed_seconds": sum(c["previous_process_elapsed_seconds"] for c in contexts.values()),
        "historical_component_process_elapsed_seconds": sum(row.get("elapsed_seconds", 0.) for row in historical),
        "new_repair_process_elapsed_seconds": new_row.get("elapsed_seconds"),
        "cost_accounting": "Original two M0 processes once, all four46615238 component processes once (including its failed reverse MZ), and this one new repair process once. Repeated context references are not additional costs. These descriptive durations never replace or duplicate coordinator allocation accounting.",
        "status": "common_gain_nested_diagnostic_completed" if complete else "incomplete_common_gain_components",
        "pooled_diagnostic_complete": complete, "historical_components": historical, "components": component_rows,
        "complete_components": [{"direction": d, "model": m} for d, m in keys if (d, m) in panels],
        "unknown_components": [{"direction": d, "model": m} for d, m in keys if (d, m) not in panels],
        "fixed_primary_participant_scores": primary, "selected_replicates": [0],
        "wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "task_versions_are_independent_replications": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in common.CLAIM_FIELDS},
        "interpretation": "One original-cohort common-gain10-through20 sensitivity. Preserve three completed components and add only the fresh reverse-MZ1500 repair. Pool both directions' actual held-out log sums over96 EI/176 ER trials per participant, then equal108-person weighting. No recovery, family, generality, template/missingness/heterogeneity robustness, interval, curtailment or full-gate verdict."}
    if complete:
        scores, totals = {}, {}
        for person in sorted(primary):
            count = 96 if primary[person]["version"] == "EI" else 176
            if any(sum(panels[d, model][person]["heldout_trials"] for d in (0, 1)) != count for model in MODELS):
                raise ValueError("Both models must score the same actual96 EI/176 ER trial totals")
            totals[person] = count
            scores[person] = {"version": primary[person]["version"], **{
                model: sum(panels[d, model][person][model + "_log_sum"] for d in (0, 1))/count for model in MODELS}}
        fixed.score_panel({"participant_scores": scores}, MODELS)
        result.update(participants=108, retained_trials_per_participant=totals,
            score_normalization="sum_of_both_heldout_log_sums_divided_by_total_actual_retained_trials_per_participant",
            common_gain_participant_scores=scores, common_gain_nested_diagnostic=fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dir", type=Path, required=True)
    parser.add_argument("--repair-job-id", type=original.parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != common.IMPLEMENTATION.resolve():
        raise ValueError("Use the fixed original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.repair_dir, args.repair_job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("phase", "status", "pooled_diagnostic_complete", "unknown_components")}, indent=2))
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
