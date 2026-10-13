#!/usr/bin/env python3
"""Pool explicit terminal MZ menus with exact original and numerical-repair lineage.

Reuse the existing nine nested contrasts. Missing, held and technical cohorts
remain unknown in fixed200; these descriptive menus select no family or rate.
The coordinator establishes terminal state separately. Selected wave arrays
are46597540 fresh-both and46597546 reverse-only, supplied explicitly at the CLI.
No polling or retries.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_mvc_convergence_repair as repair
import summarize_mz_menu_wave as wave
import summarize_mz_mvc_repaired_menu as repaired42


saved, fixed = wave.saved, wave.fixed
IMPLEMENTATION, RUNS = wave.IMPLEMENTATION, wave.RUNS
PHASE = "MZ_repaired_fixed200_menu_wave_nested_diagnostics"
REVERSE_INDICES = frozenset((3, 11, 22, 36, 61, 71, 73, 83, 86, 144, 149, 151, 152, 155,
                             163, 165, 168, 171, 172, 181, 188, 196))
ORIGINAL_INDICES = frozenset((0, 92, 105))


def array_id(value):
    if not value.isdecimal() or int(value) < 1:
        raise argparse.ArgumentTypeError("Supply a positive explicit repair array ID")
    return int(value)


def input_paths(index, fresh_array=None, reverse_array=None):
    if type(index) is not int or not 0 <= index < 200:
        raise ValueError("Use a fixed cohort index0 through199")
    for array in (fresh_array, reverse_array):
        if array is not None and (type(array) is not int or array < 1):
            raise ValueError("Repair array IDs must be positive explicit integers")
    if index == 1:
        return None
    if index == 0:
        return dict(zip(("baseline", "simple", "MVZ", "MVC"), wave.input_paths(0)))
    if index == 42:
        return dict(zip(("baseline", "MC", "MV", "MVZ", "MVC", "failed", "previous"), repaired42.INPUT_REPORTS))
    paths = dict(zip(("baseline", *saved.RIVALS), wave.input_paths(index)))
    if index in ORIGINAL_INDICES:
        return paths
    paths["failed"] = repair.failed_path(index)
    if index in (2, 3):
        directory = RUNS / "20261005-MZ-MVC-repair-cases-46597025" / f"replicate{index}"
    else:
        array = reverse_array if index in REVERSE_INDICES else fresh_array
        if array is None:
            return None
        directory = RUNS / f"20261005-MZ-MVC-repair-wave-{array}" / f"replicate{index}"
    paths["MVC"] = directory / repair.REPORT_NAME
    return paths


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Retain finite nonnegative attempt and fit costs")
    return value


def terminal_metadata(report):
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("Require dated terminal attempt provenance")
    duration(report.get("elapsed_seconds"))
    return {key: report[key] for key in ("started_utc", "finished_utc", "elapsed_seconds")}


def replay(report):
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (report.get("saved_scores_replayed") is not True
            or report.get("saved_score_replay_absolute_tolerance") != saved.REPLAY_ABSOLUTE_TOLERANCE
            or duration(discrepancy) > saved.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("Require unchanged successful baseline replay at1e-12")


def validate_repair(report, base, failed, implementation, baseline, index):
    """Validate complete or technical1500 repair metadata without phase relabeling."""
    technical = report.get("status") == "technical_rival_failure"
    job = str(report.get("job_id", ""))
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != repair.PHASE
            or report.get("generating_model") != "MZ" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != saved.cell_seed("MZ", index)
            or report.get("source_dir") is None
            or fixed.episode_path(report["source_dir"]) != repair.saved.source_directory(index).resolve()
            or report.get("source_report") != "single_process_recovery.json"
            or report.get("failed_report") != str(repair.failed_path(index))
            or not job.isdecimal() or int(job) < 1 or (index in (2, 3) and job != "46597025")):
        raise ValueError("Require the selected real repair phase, cohort, source and worker identity")
    if (any(report.get(field) != "not_evaluated" for field in repair.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_M0_MZ_refitted", "observed_choices_or_RTs_used", "old_synthetic_responses_used",
                "wave_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected"))
            or report.get("selected_replicates") != [index] or report.get("selected_models") != ["MVC"]
            or report.get("fixed_cohort_count") != 200 or report.get("component_scope") != "single_MVC_numerical_repair"):
        raise ValueError("Preserve the single-component diagnostic-only scope")
    terminal = terminal_metadata(report)
    reuse = index in REVERSE_INDICES
    cap = repair.REUSE_MAX_SECONDS if reuse else repair.MAX_SECONDS
    if type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= cap:
        raise ValueError("Retain the selected repair mode's bounded process cap")
    fits, scored = report.get("fits"), report.get("scored_primary_directions")
    if not isinstance(fits, list) or len(fits) > 2 or not isinstance(scored, list) or len(scored) > len(fits):
        raise ValueError("Require an ordered returned-fit/scored-direction prefix")
    expected_scored = [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in saved.HALVES[:len(scored)]]
    if scored != expected_scored:
        raise ValueError("Scored directions must retain their original order")
    if technical:
        if (report.get("rival_diagnostics_complete") is not False or not isinstance(report.get("error"), str)
                or not report["error"] or "participant_scores" in report
                or report.get("completed_rivals") != [] or report.get("diagnostics") != {}):
            raise ValueError("A technical repair cannot supply complete scientific scores")
    elif (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or len(fits) != 2 or len(scored) != 2 or report.get("completed_rivals") != ["MVC"]):
        raise ValueError("Only a complete two-direction repair may enter a menu")
    runtime = deepcopy(implementation)
    runtime["optimizer"]["maximum_sweeps"] = 1500
    for field, expected in (("implementation", implementation), ("runtime_implementation", runtime)):
        if (not technical or field in report) and report.get(field) != expected:
            raise ValueError("Preserve original500 and the maximum-sweeps-only1500 runtime copy")
    context_validated = report.get("repair_context_validated")
    source_validated = report.get("source_baseline_validated")
    recorded = report.get("numerical_repair")
    if context_validated is False:
        expected = {"kind": "maximum_sweeps_only", "original_maximum_sweeps": 500,
                    "runtime_maximum_sweeps": 1500, "nonconverged_checkpoint_state_reused": False,
                    "source_and_preserved_fit_validated_against_actual_cohort": False}
        if (not technical or recorded != expected or source_validated is not False or fits or scored
                or report.get("source_job_id") is not None or report.get("saved_scores_replayed") is not False
                or "cumulative_process_elapsed_seconds" in report):
            raise ValueError("An unvalidated technical checkpoint cannot claim source, fits or prior costs")
        return {**terminal, "status": report["status"], "error": report["error"], "returned_fits": []}
    if context_validated is not True or type(source_validated) is not bool:
        raise ValueError("Repair/source validation states must remain explicit")
    if failed is None:
        # Identity/claim/status checks above still apply when the selected
        # original attempt is missing. The caller retains the missing lineage
        # as unknown; no fit or score panel can be admitted without it.
        return None
    runtime, expected = repair.repair_context(index, implementation, failed)
    if expected["classification"] != (repair.SECOND if reuse else repair.FIRST):
        raise ValueError("The selected array mode differs from the diagnosed immutable original failure")
    if source_validated:
        if (failed.get("source_finished_utc") != base["finished_utc"]
                or report.get("source_finished_utc") != base["finished_utc"]
                or report.get("source_job_id") != base.get("job_id")
                or any(fit.get("participants") != sorted(baseline) for fit in failed["fits"])):
            raise ValueError("Bind both attempts to the exact baseline completion and participant identities")
        saved.completed_support(report)
        if reuse:
            fixed.validate_fit(failed["fits"][0], "MVC", *saved.HALVES[0], implementation, baseline)
    elif not technical or fits or scored or report.get("saved_scores_replayed") is not False or report.get("source_job_id") is not None:
        raise ValueError("An unvalidated source cannot supply fits or replay evidence")
    expected["source_and_preserved_fit_validated_against_actual_cohort"] = source_validated
    if not isinstance(recorded, dict):
        raise ValueError("Require reconstructed exact repair metadata")
    if reuse and fits:
        original_first = deepcopy(failed["fits"][0])
        original_first.update(repair_execution="reused_validated_original_fit", fit_maximum_sweeps=500,
                              original_fit_elapsed_seconds=failed["fits"][0]["elapsed_seconds"])
        if ({key: value for key, value in fits[0].items() if key != "elapsed_seconds"}
                != {key: value for key, value in original_first.items() if key != "elapsed_seconds"}):
            raise ValueError("Preserve every original first-fit parameter/history; only copy duration may differ")
        expected["reused_first_fit_materialization_elapsed_seconds"] = duration(fits[0].get("elapsed_seconds"))
        rescored = recorded.get("reused_first_fit_rescored")
        rescore_cost = recorded.get("reused_first_fit_rescore_elapsed_seconds")
        if type(rescored) is not bool or (not technical and not rescored) or (scored and not rescored):
            raise ValueError("Preserve successful first-fit rescoring evidence")
        if rescore_cost is not None:
            duration(rescore_cost)
        if rescored and rescore_cost is None:
            raise ValueError("Preserve separate first-fit rescoring cost")
        expected.update(reused_first_fit_rescored=rescored, reused_first_fit_rescore_elapsed_seconds=rescore_cost)
    if recorded != expected:
        raise ValueError("Recorded repair lineage, numerical scope or cost provenance differs")
    if duration(report.get("cumulative_process_elapsed_seconds")) != expected["previous_process_elapsed_seconds"] + report["elapsed_seconds"]:
        raise ValueError("Count original attempt plus new process once; never add preserved-fit cost again")
    if fits or report.get("saved_scores_replayed") is True or not technical:
        replay(report)
    returned = []
    for half, (fit, (train, held)) in enumerate(zip(fits, saved.HALVES)):
        preserved = reuse and half == 0
        if (fit.get("model") != "MVC" or fit.get("rival") != "fixed" or fit.get("training_runs") != list(train)
                or fit.get("scoring_runs") != list(held) or fit.get("participants") != sorted(baseline)
                or fit.get("repair_execution") != ("reused_validated_original_fit" if preserved else "fresh_fit")
                or fit.get("fit_maximum_sweeps") != (500 if preserved else 1500)):
            raise ValueError("Retain exact fit/half/source and numerical provenance")
        duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            fixed.validate_fit(fit, "MVC", train, held, implementation if preserved else runtime, baseline)
        elif not technical or fit.get("converged") is not False or half < len(scored):
            raise ValueError("A nonconverged fit remains technical and cannot be a scored direction")
        history = fit.get("objective_history")
        if (not isinstance(history, list) or not 2 <= len(history) <= (500 if preserved else 1500)
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in history)
                or not isinstance(fit.get("subject_optimizer_failures"), list)):
            raise ValueError("Retain the finite returned history and final subject-failure metadata")
        returned.append({**{key: fit.get(key) for key in ("model", "training_runs", "scoring_runs", "converged",
                        "elapsed_seconds", "repair_execution", "fit_maximum_sweeps", "subject_optimizer_failures")},
                        "returned_sweeps": len(history)})
    if technical:
        return {**terminal, "status": report["status"], "error": report["error"], "returned_fits": returned,
                "scored_primary_directions": scored, "numerical_repair": recorded,
                "cumulative_process_elapsed_seconds": report["cumulative_process_elapsed_seconds"]}
    scores = fixed.score_panel(report, ("M0", "MZ", "MVC"))
    if set(scores) != set(baseline) or any(scores[person][key] != baseline[person][key]
                                         for person in baseline for key in ("M0", "MZ", "version")):
        raise ValueError("Preserve every original baseline score last bit and task-version identity")
    return scores


def build_repaired_summary(base, mc, mv, mvz, mvc, failed, implementation, index):
    if mvc.get("status") != "rival_diagnostics_completed" or mvc.get("rival_diagnostics_complete") is not True:
        raise ValueError("A technical repair remains unknown and cannot enter menu contrasts")
    original_validator = saved.validate_rival

    def selected_validator(report, model, source, spec, baseline, selected_index):
        if model != "MVC":
            return original_validator(report, model, source, spec, baseline, selected_index)
        if report is not mvc or source is not base or selected_index != index or spec != implementation:
            raise ValueError("The repair admission is scoped to this exact MVC panel")
        return validate_repair(report, source, failed, spec, baseline, index)

    with patch.object(saved, "validate_rival", selected_validator):
        summary = saved.build_summary(base, mc, mv, mvz, mvc, implementation, index)
    summary.update(phase="MZ_runtime1500_repaired_menu_nested_diagnostics", fixed_cohort_count=200,
                   runtime_implementation=mvc["runtime_implementation"], numerical_repair=mvc["numerical_repair"],
                   repair_process_elapsed_seconds=mvc["elapsed_seconds"],
                   cumulative_process_elapsed_seconds=mvc["cumulative_process_elapsed_seconds"],
                   **{field: "not_evaluated" for field in repair.CLAIM_FIELDS})
    return summary


def pool(indices, implementation, fresh_array=None, reverse_array=None):
    input_paths(0, fresh_array, reverse_array)
    if (implementation["optimizer"]["maximum_sweeps"] != 500
            or implementation["recovery_plan"]["replicates_per_qualified_generating_cell"] != 200
            or len(set(indices)) != len(indices) or any(type(index) is not int or not 0 <= index < 200 for index in indices)):
        raise ValueError("Retain original500, fixed200 and distinct explicit terminal indices")
    rows, summaries, completed = [], [], []
    for index in range(200):
        row = {"replicate_index": index, "status": "unknown"}
        rows.append(row)
        if index == 1 or index not in indices:
            row["reason"] = "MC1_interrupted_retry_held" if index == 1 else "not_selected_as_terminal_by_coordinator"
            continue
        paths = input_paths(index, fresh_array, reverse_array)
        if paths is None:
            row["reason"] = "repair_array_not_declared"
            continue
        row["input_reports"] = {name: str(path) for name, path in paths.items()}
        reports, missing = {}, []
        for name, path in paths.items():
            try:
                reports[name] = json.loads(path.read_text())
            except FileNotFoundError:
                missing.append(str(path))
        base = reports.get("baseline")
        baseline = None if base is None else (fixed.validate_base(base, implementation) if index == 0
                                              else saved.validate_base(base, implementation, index))
        technical = None
        if baseline is not None:
            for model in (("simple", "MVZ", "MVC") if index == 0 else saved.RIVALS):
                if model not in reports:
                    continue
                report = reports[model]
                if index == 0:
                    fixed.validate_addon(report, ("MC", "MV") if model == "simple" else (model,), base, implementation, baseline)
                elif model != "MVC" or index in ORIGINAL_INDICES:
                    saved.validate_rival(report, model, base, implementation, baseline, index)
                elif index == 42:
                    if all(name in reports for name in ("failed", "previous")):
                        repaired42.validate_completion(report, base, implementation, baseline, reports["failed"], reports["previous"])
                else:
                    checked = validate_repair(report, base, reports.get("failed"), implementation, baseline, index)
                    if report["status"] == "technical_rival_failure":
                        technical = checked
        if missing or technical is not None:
            row.update(reason="incomplete_terminal_menu", missing_reports=missing, technical_MVC=technical)
            continue
        if index == 0:
            summary = fixed.build_summary(*(reports[name] for name in ("baseline", "simple", "MVZ", "MVC")), implementation)
        elif index in ORIGINAL_INDICES:
            summary = saved.build_summary(*(reports[name] for name in ("baseline", *saved.RIVALS)), implementation, index)
        elif index == 42:
            summary = repaired42.build_summary(*(reports[name] for name in ("baseline", *saved.RIVALS, "failed", "previous")), implementation)
        else:
            summary = build_repaired_summary(*(reports[name] for name in ("baseline", *saved.RIVALS, "failed")), implementation, index)
        row.update(status="nested_diagnostics_completed", summary=summary)
        summaries.append(summary)
        completed.append(index)
    maximal = Counter(tuple(summary["maximal_eligible_models_diagnostic"]) for summary in summaries)
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ",
            "implementation": implementation, "fixed_cohort_count": 200, "selected_terminal_indices": sorted(indices),
            "fresh_array_job_id": fresh_array, "reverse_array_job_id": reverse_array,
            "completed_indices": completed, "unknown_indices": [row["replicate_index"] for row in rows if row["status"] == "unknown"],
            "cohorts": rows, "diagnostic_counts": {"completed_menus": len(completed), "unknown_menus": 200 - len(completed),
                "maximal_eligible_model_sets_diagnostic": [{"models": list(models), "menus": count} for models, count in sorted(maximal.items())],
                "menus_with_multiple_incomparable_eligible_models": sum(summary["multiple_incomparable_eligible_models"] for summary in summaries)},
            **{field: "not_evaluated" for field in repair.CLAIM_FIELDS}, "full_recovery_pass": None, "episode_terminal": False,
            "wave_selected": False, "retry_wave_selected": False, "full_menu_wave_selected": False,
            "interpretation": "Raw counts describe complete nested-menu diagnostics only. Held, missing, unselected and technical cohorts remain unknown in fixed200, never scientific failures. No family, rate, nonnested arbitration, interval or gate verdict is supplied.",
            "process_state_boundary": "The coordinator separately establishes selected terminal processes; this pool never polls or dispatches."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-indices", type=wave.selected_indices, required=True)
    parser.add_argument("--fresh-array-job-id", type=array_id)
    parser.add_argument("--reverse-array-job-id", type=array_id)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous summaries; select a fresh output")
    result = pool(args.terminal_indices, json.loads(IMPLEMENTATION.read_text()), args.fresh_array_job_id, args.reverse_array_job_id)
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("phase", "completed_indices", "diagnostic_counts", "family_selection")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
