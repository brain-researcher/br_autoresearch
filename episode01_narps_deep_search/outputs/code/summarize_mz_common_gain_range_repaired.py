#!/usr/bin/env python3
"""Pool four explicit common-gain components after two diagnosed M0 failures.

M0 uses fresh1500 fits; previously unattempted MZ uses original500 fits.
Each direction's original failed process is retained once as shared lineage,
separate from the four new process durations and coordinator allocation costs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from qualify_assets import require_compute
import run_mz_common_gain_range_repair as repair
import summarize_mz_common_gain_range as original
import summarize_mz_fixed_menu as fixed


common = original.common
MODELS = ("M0", "MZ")
PHASE = "MZ0_common_gain_after_M0_exhaustion_four_component_nested_diagnostic"
SOURCE_REPORT = common.SOURCE / common.SOURCE_REPORT


def report_paths(source_root, job_id):
    job = original.parent_job(job_id)
    root = Path(source_root).resolve()
    if root != (common.SOURCE.parent / f"20261005-MZ-common-gain-repair-{job}").resolve():
        raise ValueError("Use the explicit selected four-component common-gain repair parent")
    return {(direction, model): root / f"dir{direction}" / model / repair.REPORT_NAME
            for direction in (0, 1) for model in MODELS}


def validate_component(report, direction, model, job_id, spec, primary, source, context):
    train, held = common.selected_runs(direction)
    runtime = repair.runtime_specification(spec, model)
    technical = report.get("status") == "technical_common_gain_failure"
    expected_kind = "M0_numerical_repair" if model == "M0" else "previously_unattempted_MZ_component"
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != repair.PHASE
            or report.get("generating_model") != "MZ" or type(report.get("replicate_index")) is not int
            or report["replicate_index"] != 0 or report.get("seed") != common.shared.SEED
            or type(report.get("direction")) is not int or report["direction"] != direction
            or report.get("fitted_model") != model or report.get("fitted_models") != [model]
            or report.get("rival") != "fixed" or str(report.get("job_id")) != str(job_id)
            or report.get("training_runs") != list(train) or report.get("scoring_runs") != list(held)
            or report.get("execution_kind") != expected_kind
            or report.get("failed_report") != context["failed_report"]
            or report.get("source_dir") != str(common.SOURCE)
            or report.get("source_report") != str(SOURCE_REPORT)
            or report.get("gain_range_dollars") != [10, 20] or report.get("filter_boundary_inclusive") is not True
            or report.get("score_normalization") != common.NORMALIZATION):
        raise ValueError("Preserve each actual selected single-model/direction/source and execution kind")
    if (report.get("numerical_override") != repair.numerical_override(model)
            or report.get("original_maximum_sweeps") != 500
            or report.get("fitted_maximum_sweeps") != runtime["optimizer"]["maximum_sweeps"]
            or any(report.get(field) != "not_evaluated" for field in common.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("selected_replicates") != [0]
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_common_gain", "observed_choices_or_RTs_used", "wave_selected",
                "retry_wave_selected", "automatic_retry", "task_versions_are_independent_replications", "episode_terminal"))
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= repair.MAX_SECONDS
            or "cumulative_process_elapsed_seconds" in report):
        raise ValueError("Retain model-specific numerical settings, closed claims and shared-cost reference semantics")
    if "implementation" in report:
        if (report["implementation"] != spec or report.get("original_implementation") != spec
                or report.get("runtime_implementation") != runtime):
            raise ValueError("Preserve original500 source and distinct actual fit specification")
    elif (not technical or report.get("original_implementation") is not None
            or report.get("runtime_implementation") is not None):
        raise ValueError("Only pre-specification technical checkpoints may omit loaded specifications")
    elapsed = original.duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Read only explicit terminal component attempts") from error
    contextual, validated, replayed = (report.get(field) for field in
        ("source_context_validated", "source_baseline_validated", "saved_primary_scores_replayed"))
    if any(type(value) is not bool for value in (contextual, validated, replayed)):
        raise ValueError("Source/context/replay qualification must remain explicit")
    if contextual:
        if (report.get("prior_failed_context") != context or report.get("implementation") != spec
                or report.get("prior_direction_process_elapsed_seconds") != context["previous_process_elapsed_seconds"]):
            raise ValueError("Reconstruct the exact original direction failure as shared lineage")
    elif (report.get("prior_failed_context") is not None
            or report.get("prior_direction_process_elapsed_seconds") is not None or validated or replayed):
        raise ValueError("Unqualified context cannot supply source/replay or prior-cost evidence")
    if validated:
        if (not contextual or report.get("source_job_id") != source["job_id"]
                or report.get("source_finished_utc") != source["finished_utc"]
                or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]):
            raise ValueError("The selected source must remain the exact original full108-person cohort")
    elif not technical or replayed or report.get("source_job_id") is not None or report.get("source_finished_utc") is not None:
        raise ValueError("Unvalidated sources cannot supply qualified identities or replay")
    if replayed:
        difference = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_primary_participant_scores") != primary
                or report.get("saved_score_replay_absolute_tolerance") != common.shared.REPLAY_ABSOLUTE_TOLERANCE
                or not original.finite(difference) or not 0 <= difference <= common.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("Preserve original primary last bits and exact replay tolerance")
    elif not technical or "saved_primary_participant_scores" in report:
        raise ValueError("New fits require original full-range primary replay")
    fits, scored, completed, calls = (report.get(key) for key in
        ("fits", "scored_fits", "completed_model_scores", "fresh_fit_calls"))
    expected = [{"model": model, "training_runs": list(train), "scoring_runs": list(held)}]
    if (not isinstance(fits, list) or len(fits) > 1 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored) + 1
            or scored != expected[:len(scored)] or not isinstance(completed, dict)
            or set(completed) != ({model} if scored else set())
            or type(calls) is not int or calls not in (0, 1) or not len(fits) <= calls <= len(fits) + 1
            or ((calls or fits or completed) and not (contextual and validated and replayed))):
        raise ValueError("Retain at most one fresh fit and its exact successful score prefix")
    support = report.get("retained_support")
    if support is not None:
        if not replayed or support != context["retained_support"]:
            raise ValueError("Use exactly the previously validated filtered trial support")
        original.validate_support(support, primary, train, held)
    elif not technical or calls or fits or completed:
        raise ValueError("Fitting requires the exact complete filtered support")
    row = {"direction": direction, "fitted_model": model, "job_id": str(job_id), "phase": report["phase"],
        "execution_kind": expected_kind, "elapsed_seconds": elapsed, "finished_utc": report["finished_utc"],
        "original_maximum_sweeps": 500, "fitted_maximum_sweeps": runtime["optimizer"]["maximum_sweeps"],
        "source_context_validated": contextual, "prior_failure_direction_reference": direction,
        "fresh_fit_calls": calls}
    if technical:
        if (report.get("common_gain_component_complete") is not False or "participant_scores" in report
                or not isinstance(report.get("error"), str) or not report["error"]):
            raise ValueError("Technical components remain unknown without final scores")
        row.update(status="technical_unknown", error=report["error"])
    elif (report.get("status") != "common_gain_component_completed" or report.get("common_gain_component_complete") is not True
            or not (contextual and validated and replayed) or len(fits) != 1 or calls != 1 or scored != expected):
        raise ValueError("A completed model component requires one valid fresh fit and all participant scores")
    else:
        row["status"] = "complete_component"
    returned = []
    for fit in fits:
        if (fit.get("model") != model or fit.get("rival") != "fixed" or fit.get("participants") != sorted(primary)
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)):
            raise ValueError("Retain the selected training-only fit identity")
        consumed = original.duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            common.validate_fit(fit, model, train, held, runtime, primary)
        elif fit.get("converged") is not False or scored or not technical:
            raise ValueError("Only a converged model fit may supply scored evidence")
        returned.append({"converged": fit["converged"], "elapsed_seconds": consumed})
    if completed:
        panel = completed[model]
        if not isinstance(panel, dict) or set(panel) != set(primary):
            raise ValueError("A completed score prefix requires all108 source participants")
        for person, value in panel.items():
            if (set(value) != {"log_sum", "trials"} or not original.finite(value["log_sum"])
                    or type(value["trials"]) is not int or value["trials"] != support[person]["heldout_trials"]):
                raise ValueError("Retain finite log sums and actual held-out counts")
    if technical:
        row.update(returned_fits=returned, scored_fits=scored, completed_model_scores=completed)
        return row, None, support
    scores = report.get("participant_scores")
    if not isinstance(scores, dict) or set(scores) != set(primary):
        raise ValueError("A completed model requires108 final score rows")
    for person, score in scores.items():
        if (set(score) != {"version", "heldout_trials", model + "_log_sum"}
                or score["version"] != primary[person]["version"]
                or type(score["heldout_trials"]) is not int or score["heldout_trials"] != support[person]["heldout_trials"]
                or not original.finite(score[model + "_log_sum"])
                or score[model + "_log_sum"] != completed[model][person]["log_sum"]):
            raise ValueError("Final single-model sums/counts must equal the preserved scoring evidence")
    return row, scores, support


def pool(source_root, job_id, spec):
    paths = report_paths(source_root, job_id)
    if spec["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Retain original500 source and MZ fitting")
    source = json.loads(SOURCE_REPORT.read_text())
    primary = fixed.validate_base(source, spec)
    contexts = {}
    for direction in (0, 1):
        failed = json.loads(repair.failed_path(direction).read_text())
        contexts[direction] = repair.repair_context(failed, source, primary, spec, direction)
    for person in primary:
        a, b = (contexts[direction]["retained_support"][person] for direction in (0, 1))
        if ({tuple(slot) for slot in a["training_slots"]} != {tuple(slot) for slot in b["heldout_slots"]}
                or {tuple(slot) for slot in a["heldout_slots"]} != {tuple(slot) for slot in b["training_slots"]}):
            raise ValueError("The original direction failures must retain complementary filtered trials")
    rows, panels = [], {}
    for (direction, model), path in paths.items():
        try:
            report = json.loads(path.read_text())
        except FileNotFoundError:
            rows.append({"direction": direction, "fitted_model": model, "status": "missing_unknown", "report": str(path)})
            continue
        row, scores, support = validate_component(report, direction, model, job_id, spec, primary, source, contexts[direction])
        row["report"] = str(path)
        rows.append(row)
        if scores is not None:
            panels[direction, model] = scores
    complete = len(panels) == 4
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": common.shared.SEED,
        "source_root": str(Path(source_root).resolve()), "job_id": str(job_id),
        "source_dir": str(common.SOURCE), "source_report": str(SOURCE_REPORT),
        "source_job_id": source["job_id"], "source_finished_utc": source["finished_utc"],
        "original_implementation": spec, "runtime_implementations": {model: repair.runtime_specification(spec, model) for model in MODELS},
        "per_model_direction_maximum_sweeps": {"M0": [1500, 1500], "MZ": [500, 500]},
        "gain_range_dollars": [10, 20], "filter_boundary_inclusive": True,
        "prior_direction_failures": contexts,
        "unique_prior_process_elapsed_seconds": sum(context["previous_process_elapsed_seconds"] for context in contexts.values()),
        "new_component_process_elapsed_seconds": sum(row.get("elapsed_seconds", 0.) for row in rows),
        "cost_accounting": "The two original failure processes occur once each in prior_direction_failures. Each new model process has its own elapsed time; these descriptive durations do not recharge or replace coordinator allocation accounting. No per-component prior-plus-new cumulative cost is used.",
        "status": "common_gain_nested_diagnostic_completed" if complete else "incomplete_common_gain_components",
        "pooled_diagnostic_complete": complete,
        "complete_components": [{"direction": direction, "model": model} for direction, model in paths if (direction, model) in panels],
        "unknown_components": [{"direction": direction, "model": model} for direction, model in paths if (direction, model) not in panels],
        "components": rows, "fixed_primary_participant_scores": primary,
        "selected_replicates": [0], "wave_selected": False, "retry_wave_selected": False, "automatic_retry": False,
        "task_versions_are_independent_replications": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in common.CLAIM_FIELDS},
        "interpretation": "One saved-cohort common-gain10-through20 sensitivity. M0 is freshly fitted under1500 after its exact500 failures; MZ is newly fitted under original500. Both directions and both models must complete. Per-person log sums use all retained96 EI or176 ER trials, followed by equal108-participant weighting. Original failure contacts and fixed-primary scores remain separate. No recovery, family, generality, template/missingness/heterogeneity robustness, interval, curtailment or gate verdict."}
    if complete:
        scores, totals = {}, {}
        for person in sorted(primary):
            counts = {model: sum(panels[direction, model][person]["heldout_trials"] for direction in (0, 1)) for model in MODELS}
            expected = 96 if primary[person]["version"] == "EI" else 176
            if any(count != expected for count in counts.values()):
                raise ValueError("Both models must exhaust the same96 EI or176 ER held-out trial slots")
            totals[person] = expected
            scores[person] = {"version": primary[person]["version"],
                **{model: sum(panels[direction, model][person][model + "_log_sum"] for direction in (0, 1)) / expected for model in MODELS}}
        fixed.score_panel({"participant_scores": scores}, MODELS)
        result.update(participants=108, retained_trials_per_participant=totals,
            score_normalization="sum_of_both_heldout_log_sums_divided_by_total_actual_retained_trials_per_participant",
            common_gain_participant_scores=scores, common_gain_nested_diagnostic=fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-id", type=original.parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != common.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.source_root, args.job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "complete_components", "unknown_components", "pooled_diagnostic_complete")}, indent=2), flush=True)
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
