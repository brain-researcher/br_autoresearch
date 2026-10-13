#!/usr/bin/env python3
"""Pool explicit terminal null menus with fixed200 support and unknowns retained.

Use only the selected fixed reports and existing nine-contrast/partial-order
builder. Neither a unique maximal model nor these descriptive counts select a
family, estimate a false-selection rate, authorize a retry or pass recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import summarize_mz_null_menu as menu
import summarize_mz_null_repaired_menu as pilot_menu
import summarize_mz_null_baseline_wave as baseline
import run_mz_null_indexed_simple_rival as simple
import run_mz_null_indexed_rival as mvz_runner
import run_mz_null_indexed_mvc as mvc_runner
import run_mz_null_completed_baseline_rival as completed35
import run_mz_null_additional_completed_rival as additional
import mz_null_mvc_repair_admission as mvc_repairs
import mz_null_mvc_reverse_admission as mvc_reverses


IMPLEMENTATION = menu.IMPLEMENTATION
RUNS = baseline.RUNS
PHASE = "M0_null_indexed_menu_wave_nested_diagnostics"
UNADMITTED = frozenset((108, 176))
CLAIM_FIELDS = (*menu.null.CLAIM_FIELDS, "early_kill")
HALVES = menu.HALVES
FIXED_COHORTS = 200


def validate_additional_array(array):
    if array is not None and (type(array) is not int or array < 1):
        raise ValueError("Supply an explicit positive additional MVC array job ID")


def input_paths(index, additional_mvc_array_job_id=None, mvc_repair_arrays=None, mvc_reverse_arrays=None):
    baseline.indexed.null_seed(index)
    validate_additional_array(additional_mvc_array_job_id)
    repair_mapping = mvc_repairs.validate_mapping(mvc_repair_arrays)
    reverse_mapping = mvc_reverses.validate_mapping(mvc_reverse_arrays)
    if index in UNADMITTED:
        if additional_mvc_array_job_id is None:
            raise ValueError("Null108/176 admission requires an explicit additional MVC array job ID")
        directory, completion_report = additional.source_paths(index)
        paths = {"baseline": completion_report, "failed_baseline": directory / baseline.indexed.REPORT_NAME}
        for model in menu.RIVALS:
            component = (RUNS / f"20261005-M0-null-MVC-additional-array-{additional_mvc_array_job_id}" / f"replicate{index}"
                         if model == "MVC" else RUNS / f"20261005-M0-null{index}-{model}-completed-baseline-46595106")
            paths[model] = component / additional.REPORT_NAME
        return paths
    if index == 0:
        return dict(zip(("baseline", *menu.RIVALS, "failed_MVC"), pilot_menu.INPUT_REPORTS))
    directory = simple.source_directory(index)
    paths = {"baseline": completed35.COMPLETION_REPORT if index == 35 else directory / baseline.indexed.REPORT_NAME}
    for model in menu.RIVALS:
        if index == 35 and model != "MVC":
            paths[model] = RUNS / f"20261005-M0-null35-{model}-completed-baseline-46589557" / completed35.REPORT_NAME
        else:
            if model in ("MC", "MV"):
                array = 46583757 if 29 <= index <= 32 else 46584095
                name = simple.REPORT_NAME
            elif model == "MVZ":
                array = 46581137 if index <= 4 else (46584696 if 29 <= index <= 32 else 46585182)
                name = mvz_runner.REPORT_NAME
            else:
                array = 46591849 if index == 1 else 46592261
                name = mvc_runner.REPORT_NAME
            paths[model] = RUNS / f"20261005-M0-null-{model}-indexed-array-{array}" / f"replicate{index}" / name
    if index == 35:
        paths["failed_baseline"] = completed35.FAILED_REPORT
    replacement = mvc_repairs.input_paths(index, repair_mapping)
    if replacement is not None:
        paths.update(replacement)
    reverse = mvc_reverses.input_paths(index, reverse_mapping)
    if reverse is not None:
        paths.update(reverse)
    return paths


def terminal_metadata(report):
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("A selected terminal attempt requires explicit dated provenance")
    value = report.get("elapsed_seconds")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("A terminal technical attempt requires finite nonnegative consumed process cost")
    return {field: report[field] for field in ("started_utc", "finished_utc", "elapsed_seconds")}


def validate_baseline(base, truth, people, index, implementation, failed=None):
    if index in UNADMITTED:
        return additional.validate_completed_baseline(index, base, failed, truth, people, implementation)
    if index == 35:
        return completed35.validate_completed_baseline(base, failed, truth, people, implementation)
    baseline.completed_diagnostic(base, truth, people, index, implementation)
    return None


def technical_completed_baseline(base, failed, truth, people, implementation, index=35):
    """Retain a technical completion checkpoint without counting a baseline."""
    if index in UNADMITTED:
        # Reuse this exact completion check under the same isolated identities
        # used by the additional completion and its successful-source validator.
        if base.get("automatic_retry") is not False or base.get("early_kill") != "not_evaluated":
            raise ValueError("Additional technical completions cannot select a retry or early-kill verdict")
        with additional.selected_baseline(index):
            return technical_completed_baseline(base, failed, truth, people, implementation)
    preserved, metadata = completed35.completion.completion_context(failed, truth, people, implementation)
    if (base.get("analysis_id") != "behavioral_mechanism_kill_test" or base.get("phase") != completed35.completion.PHASE
            or base.get("generating_model") != "M0" or base.get("replicate_index") != completed35.completion.REPLICATE_INDEX
            or base.get("seed") != completed35.completion.SEED or str(base.get("job_id")) != completed35.COMPLETION_JOB_ID
            or base.get("source_dir") != str(completed35.completion.SOURCE)
            or base.get("source_report") != completed35.completion.SOURCE_REPORT
            or str(base.get("source_job_id")) != completed35.completion.SOURCE_JOB_ID
            or base.get("implementation", implementation) != implementation
            or any(base.get(field) != "not_evaluated" for field in menu.null.CLAIM_FIELDS)
            or base.get("full_recovery_pass") is not None or base.get("episode_terminal") is not False
            or any(base.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                   "saved_completed_fits_refitted", "unfinished_fit_state_reused", "wave_selected", "retry_wave_selected", "full_menu_wave_selected"))):
        raise ValueError("A technical completion must preserve its exact failed-source lineage and scope")
    if base.get("source_context_validated") is True:
        if any(base.get(key) != value for key, value in metadata.items()) or base.get("fits", [])[:3] != preserved:
            raise ValueError("The technical completion must retain its exact validated three-fit prefix and costs")
    elif base.get("source_context_validated") is not False or base.get("fits") != []:
        raise ValueError("An unvalidated completion cannot claim preserved fits")
    terminal_metadata(base)
    return baseline.technical_metadata(base)


def expected_override():
    return {"kind": "maximum_sweeps_only", "original_maximum_sweeps": 500, "runtime_maximum_sweeps": 1000,
            "fresh_fit_directions": [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES],
            "initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
            "nonconverged_checkpoint_state_reused": False,
            "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
            "qualifying_null0_report": str(mvc_runner.QUALIFYING_NULL0_REPORT),
            "qualification_boundary": "Coordinator records terminal representative success; this runner does not reread or refit that pilot"}


def rival_identity(report, model, index, base, implementation, lineage):
    technical = report.get("status") == "technical_rival_failure"
    directory = menu.null.SOURCE if index == 0 else simple.source_directory(index)
    phase = (pilot_menu.repair.PHASE if model == "MVC" else "M0_null_saved_representative_rival_diagnostics") if index == 0 else (
        mvc_runner.PHASE if model == "MVC" else (completed35.PHASE if index == 35 else
        (simple.PHASE if model in ("MC", "MV") else mvz_runner.PHASE)))
    source_report = (menu.null.SOURCE_REPORT if index == 0 else
        (str(completed35.COMPLETION_REPORT) if index == 35 else
         (str(directory / baseline.indexed.REPORT_NAME) if model == "MVC" else baseline.indexed.REPORT_NAME)))
    if index in UNADMITTED:
        phase = additional.PHASE
        source_report = str(additional.source_paths(index)[1])
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != phase
            or report.get("generating_model") != "M0" or report.get("fitted_model") != model
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != baseline.indexed.null_seed(index)
            or report.get("source_dir") is None or menu.fixed.episode_path(report["source_dir"]) != directory.resolve()
            or report.get("source_report") != source_report):
        raise ValueError("The rival must retain its exact real phase, model, cohort, seed and selected source")
    job = str(report.get("job_id", ""))
    exact_job = ({"MC": "46577173", "MV": "46577173", "MVZ": "46575931", "MVC": "46582236"}[model]
                 if index == 0 else ("46589557" if index == 35 and model != "MVC" else None))
    if index in UNADMITTED and model != "MVC":
        exact_job = "46595106"
    if not job.isdecimal() or int(job) < 1 or (exact_job is not None and job != exact_job):
        raise ValueError("Retain the selected component's actual worker identity")
    if (not technical or "implementation" in report) and report.get("implementation") != implementation:
        raise ValueError("The original500 implementation differs")
    source_validated = report.get("source_baseline_validated") is True if index else report.get("saved_scores_replayed") is True
    if "source_finished_utc" in report and report["source_finished_utc"] != base["finished_utc"]:
        raise ValueError("The rival's recorded source completion differs from this baseline")
    if index == 0 or not technical or source_validated:
        if str(report.get("source_job_id")) != str(base["job_id"]):
            raise ValueError("The rival's validated source worker differs from this baseline")
        # Older null.run writes the source completion only after replay. Its
        # indexed writer already knows the worker after source validation, so
        # a replay exception legitimately retains that worker but no timestamp.
        if "source_finished_utc" not in report and not (technical and report.get("saved_scores_replayed") is False):
            raise ValueError("A replayed/completed rival requires its exact source completion timestamp")
    elif index and (report.get("source_baseline_validated") is not False or report.get("source_job_id") is not None):
        raise ValueError("An unvalidated technical source must remain explicitly unknown")
    if (any(report.get(field) != "not_evaluated" for field in menu.null.CLAIM_FIELDS)
            or report.get("early_kill", "not_evaluated") != "not_evaluated"
            or any(report.get(field) is not False for field in
                   ("simulation_regenerated", "saved_M0_MZ_refitted", "observed_choices_or_RTs_used", "old_synthetic_responses_used"))
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False):
        raise ValueError("A component cannot supply regeneration, family, rate, interval or gate claims")
    if index:
        replacement = index in UNADMITTED or (index == 35 and model != "MVC")
        cap = 8400 if model == "MVC" else (3300 if model == "MVZ" else 2400)
        if type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= cap:
            raise ValueError("Preserve this component's recorded bounded process cap")
        if (report.get("selected_replicates") != [index] or report.get("selected_models") != [model]
                or report.get("fixed_cohort_count") != 200 or report.get("component_scope") != f"{model}_only"
                or report.get("full_menu_wave_selected") is not False
                or report.get("wave_selected") is not (not replacement)):
            raise ValueError("Preserve the single selected component and fixed200 support")
        if replacement:
            if report.get("retry_wave_selected") is not False or report.get("simulation_copied") is not False:
                raise ValueError("Retain the completed-baseline replacement's scope")
        elif report.get("wave_scope") != f"{model}_component_only" or report.get("retry_selected") is not False:
            raise ValueError("Retain indexed component-only wave scope without retry selection")
        if (replacement or model == "MVC") and source_validated and report.get("baseline_lineage") != lineage:
            raise ValueError("Preserve exact completion lineage or the ordinary source's null lineage")
        if (replacement or model == "MVC") and report.get("simulation_copied") is not False:
            raise ValueError("The original source simulation must remain in place")
        if index in UNADMITTED and (report.get("wave_scope") != "single_completed_baseline_component"
                or report.get("retry_selected") is not False or report.get("automatic_retry") is not False
                or report.get("baseline_validation_maximum_sweeps") != 500
                or report.get("fitted_maximum_sweeps") != (1000 if model == "MVC" else 500)):
            raise ValueError("The additional component must retain original500 baseline and its exact fitted specification")
    elif report.get("wave_selected") is not False:
        raise ValueError("Preserve the original single-cohort null0 scope")
    if index and model == "MVC":
        if report.get("numerical_override") != expected_override() or report.get("automatic_retry") is not False:
            raise ValueError("Only the recorded fresh-both-halves maximum-sweeps override is allowed")
        if (not technical or "runtime_implementation" in report) and report.get("runtime_implementation") != mvc_runner.runtime_specification(implementation):
            raise ValueError("Runtime1000 may differ from original500 only at maximum_sweeps")
    return source_validated


def validate_replay(report):
    discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
    if (report.get("saved_scores_replayed") is not True
            or report.get("saved_score_replay_absolute_tolerance") != menu.null.shared.REPLAY_ABSOLUTE_TOLERANCE
            or type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
            or not 0 <= discrepancy <= menu.null.shared.REPLAY_ABSOLUTE_TOLERANCE):
        raise ValueError("Successful original baseline replay at the unchanged1e-12 tolerance is required")


def fit_spec(implementation, index, model, position):
    return mvc_runner.runtime_specification(implementation) if model == "MVC" and (index or position == 1) else implementation


def technical_rival(report, model, index, implementation, people):
    if (report.get("rival_diagnostics_complete") is not False or not isinstance(report.get("error"), str)
            or not report["error"] or "participant_scores" in report or report.get("completed_rivals") != []
            or report.get("diagnostics") != {}):
        raise ValueError("A technical component must retain its error without successful scores or claims")
    metadata = terminal_metadata(report)
    fits, scored = report.get("fits"), report.get("scored_primary_directions")
    if (not isinstance(fits, list) or len(fits) > 2 or not isinstance(scored, list) or len(scored) > len(fits)
            or scored != [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES[:len(scored)]]):
        raise ValueError("Technical checkpoints must retain an ordered returned/scored fit prefix")
    if fits or report.get("saved_scores_replayed") is True:
        validate_replay(report)
    returned = []
    for position, (fit, (train, held)) in enumerate(zip(fits, HALVES)):
        if (not isinstance(fit, dict) or fit.get("model") != model or fit.get("rival") != "fixed"
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
                or fit.get("participants") != sorted(people)):
            raise ValueError("Technical returned fits must retain the actual cohort and model/direction identity")
        elapsed = fit.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Preserve each returned technical fit's finite nonnegative consumed cost")
        if fit.get("converged") is True:
            menu.fixed.validate_fit(fit, model, train, held, fit_spec(implementation, index, model, position), people)
        elif fit.get("converged") is not False or position < len(scored):
            raise ValueError("An unvalidated fit cannot be a scored direction")
        returned.append({**{key: fit.get(key) for key in ("model", "training_runs", "scoring_runs", "converged",
                         "elapsed_seconds", "subject_optimizer_failures")},
                         "returned_sweeps": len(fit["objective_history"]) if isinstance(fit.get("objective_history"), list) else None})
    return {**metadata, "status": report["status"], "error": report["error"],
            "scored_primary_directions": scored, "returned_fits": returned}


def validate_indexed_rival(report, model, index, base, implementation, scores, people, lineage):
    rival_identity(report, model, index, base, implementation, lineage)
    menu.completed_support(report)
    if (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or report.get("source_baseline_validated") is not True or report.get("completed_rivals") != [model]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in HALVES]):
        raise ValueError("A successful rival requires both complete validated scored directions")
    validate_replay(report)
    panel = menu.fixed.score_panel(report, ("M0", "MZ", model))
    if set(panel) != set(scores) or any(panel[person][key] != scores[person][key]
                                      for person in scores for key in ("M0", "MZ", "version")):
        raise ValueError("Preserve original participant/task-version identity and every baseline score last bit")
    spec = fit_spec(implementation, index, model, 0)
    menu.fixed.fit_panel(report, [(model, train, held) for train, held in HALVES], spec, people)
    return panel


def build_indexed_summary(base, rivals, implementation, truth, people, index, failed=None,
                          additional_mvc_array_job_id=None, mvc_repair_arrays=None, failed_mvc=None,
                          mvc_reverse_arrays=None):
    if index == 0:
        raise ValueError("Indexed admission excludes the separately validated pilot0")
    repair_mapping = mvc_repairs.validate_mapping(mvc_repair_arrays)
    reverse_mapping = mvc_reverses.validate_mapping(mvc_reverse_arrays)
    input_paths(index, additional_mvc_array_job_id, repair_mapping, reverse_mapping)
    lineage = validate_baseline(base, truth, people, index, implementation, failed)

    def source_validator(source, source_truth, actual_people, spec):
        if source is not base or source_truth is not truth or actual_people is not people or spec != implementation:
            raise ValueError("The source adapter is scoped to the current indexed cohort")
        # Actual source validation already succeeded above; never relabel its phase.

    def rival_validator(report, model, source, spec, scores):
        if report is not rivals[model] or source is not base or spec != implementation:
            raise ValueError("The rival adapter is scoped to the current indexed cohort")
        if model == "MVC" and (index in repair_mapping or index in reverse_mapping):
            if (report.get("status") != "rival_diagnostics_completed"
                    or report.get("rival_diagnostics_complete") is not True or failed_mvc is None):
                raise ValueError("Only a complete repair with its exact prior failure may enter menu contrasts")
            validator = mvc_reverses.validate_completion if index in reverse_mapping else mvc_repairs.validate_repair
            return validator(report, failed_mvc, source, spec, scores, people, index)
        return validate_indexed_rival(report, model, index, source, spec, scores, people, lineage)

    with patch.object(menu.null, "validate_source", source_validator), \
            patch.object(menu.null, "REPLICATE_INDEX", index), \
            patch.object(menu.null, "SEED", baseline.indexed.null_seed(index)), \
            patch.object(menu.null, "SOURCE", simple.source_directory(index)), \
            patch.object(menu, "validate_rival", rival_validator):
        result = menu.build_summary(base, *(rivals[model] for model in menu.RIVALS), implementation, truth, people)
    result.update(phase="M0_null_indexed_complete_menu_nested_diagnostics", selected_replicates=[index],
                  fixed_cohort_count=200, baseline_lineage=lineage,
                  mvc_runtime_implementation=rivals["MVC"]["runtime_implementation"],
                  mvc_numerical_override=rivals["MVC"]["numerical_override"],
                  full_menu_wave_selected=False, retry_selected=False, early_kill="not_evaluated")
    if index in UNADMITTED:
        result["additional_mvc_array_job_id"] = additional_mvc_array_job_id
    if index in repair_mapping:
        result.update(mvc_repair_array_job_id=repair_mapping[index],
                      mvc_numerical_repair=rivals["MVC"]["numerical_repair"],
                      mvc_repair_process_elapsed_seconds=rivals["MVC"]["elapsed_seconds"],
                      cumulative_mvc_process_elapsed_seconds=rivals["MVC"]["cumulative_process_elapsed_seconds"])
    if index in reverse_mapping:
        result.update(mvc_reverse_array_job_id=reverse_mapping[index],
                      mvc_numerical_repair=rivals["MVC"]["numerical_repair"],
                      mvc_per_half_provenance=rivals["MVC"]["per_half_provenance"],
                      mvc_first_half_copy_elapsed_seconds=rivals["MVC"]["first_half_copy_elapsed_seconds"],
                      mvc_first_half_rescore_elapsed_seconds=rivals["MVC"]["first_half_rescore_elapsed_seconds"],
                      mvc_repair_process_elapsed_seconds=rivals["MVC"]["elapsed_seconds"],
                      cumulative_mvc_process_elapsed_seconds=rivals["MVC"]["cumulative_process_elapsed_seconds"])
    return result


def pool(indices, implementation, additional_mvc_array_job_id=None, mvc_repair_arrays=None, mvc_reverse_arrays=None):
    mvc_runner.runtime_specification(implementation)
    validate_additional_array(additional_mvc_array_job_id)
    repair_mapping = mvc_repairs.validate_mapping(mvc_repair_arrays)
    reverse_mapping = mvc_reverses.validate_mapping(mvc_reverse_arrays)
    selected = set(indices)
    if len(selected) != len(indices) or any(type(index) is not int or not 0 <= index < 200 for index in indices):
        raise ValueError("Supply distinct fixed cohort indices0 through199")
    cohorts, completed = [], []
    for index in range(200):
        row = {"replicate_index": index, "seed": baseline.indexed.null_seed(index), "status": "unknown"}
        cohorts.append(row)
        if index not in selected:
            row["reason"] = "not_selected_as_terminal_by_coordinator"
            continue
        if index in UNADMITTED and additional_mvc_array_job_id is None:
            row["reason"] = "baseline_completion_and_downstream_components_not_admitted"
            continue
        paths = input_paths(index, additional_mvc_array_job_id, repair_mapping, reverse_mapping)
        row["input_reports"] = {key: str(path) for key, path in paths.items()}
        try:
            base = json.loads(paths["baseline"].read_text())
        except FileNotFoundError:
            row.update(reason="selected_terminal_baseline_missing", missing_reports=[str(paths["baseline"])])
            continue
        if base.get("status") == "technical_representative_failure" and "failed_baseline" not in paths:
            baseline.validate_identity(base, index, implementation)
            terminal_metadata(base)
            row.update(reason="technical_baseline_failure", technical_baseline=baseline.technical_metadata(base))
            continue
        directory = menu.null.SOURCE if index == 0 else simple.source_directory(index)
        try:
            truth = json.loads((directory / "generating_parameters.json").read_text())
            people = menu.null.shared.load_cohort(directory / "simulated_events.csv.gz")
            failed = json.loads(paths["failed_baseline"].read_text()) if "failed_baseline" in paths else None
        except FileNotFoundError as error:
            row.update(reason="selected_terminal_source_material_missing", missing_reports=[str(error.filename or error)])
            continue
        if "failed_baseline" in paths and base.get("status") == "technical_baseline_completion_failure":
            row.update(reason="technical_baseline_completion_failure",
                       technical_baseline=technical_completed_baseline(base, failed, truth, people, implementation, index))
            continue
        lineage = validate_baseline(base, truth, people, index, implementation, failed)
        scores = menu.fixed.score_panel(base, ("M0", "MZ"))
        rivals, missing, technical = {}, [], {}
        prior_mvc = None
        if "failed_MVC" in paths:
            try:
                prior_mvc = json.loads(paths["failed_MVC"].read_text())
            except FileNotFoundError:
                missing.append(str(paths["failed_MVC"]))
        for model in menu.RIVALS:
            try:
                report = json.loads(paths[model].read_text())
            except FileNotFoundError:
                missing.append(str(paths[model]))
                continue
            if model == "MVC" and (index in repair_mapping or index in reverse_mapping):
                validator = mvc_reverses.validate_completion if index in reverse_mapping else mvc_repairs.validate_repair
                checked = validator(report, prior_mvc, base, implementation, scores, people, index)
                if report.get("status") == "technical_rival_failure":
                    if checked is not None:
                        technical[model] = {"report": str(paths[model]), **checked}
                else:
                    rivals[model] = report
                continue
            rival_identity(report, model, index, base, implementation, lineage)
            if report.get("status") == "technical_rival_failure":
                technical[model] = {"report": str(paths[model]), **technical_rival(report, model, index, implementation, people)}
            else:
                if report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True:
                    raise ValueError("A selected terminal rival must be complete or an identified technical failure")
                if index:
                    validate_indexed_rival(report, model, index, base, implementation, scores, people, lineage)
                elif model != "MVC":
                    menu.validate_rival(report, model, base, implementation, scores)
                elif prior_mvc is not None:
                    pilot_menu.validate_repaired_mvc(report, base, implementation, scores, prior_mvc, people)
                rivals[model] = report
        if missing or technical:
            row.update(reason="incomplete_terminal_menu", missing_reports=missing, technical_components=technical)
            continue
        if index == 0:
            summary = pilot_menu.build_summary(base, *(rivals[model] for model in menu.RIVALS), prior_mvc, implementation, truth, people)
        else:
            summary = build_indexed_summary(base, rivals, implementation, truth, people, index, failed,
                                            additional_mvc_array_job_id, repair_mapping, prior_mvc, reverse_mapping)
        row.update(status="complete_menu_diagnostic", summary=summary)
        completed.append(index)
    complete_rows = [row for row in cohorts if row["status"] == "complete_menu_diagnostic"]
    return {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "M0",
            "implementation": implementation, "fixed_cohort_count": 200, "selected_terminal_indices": sorted(selected),
            "additional_mvc_array_job_id": additional_mvc_array_job_id,
            "mvc_repair_arrays": {str(index): repair_mapping[index] for index in sorted(repair_mapping)},
            "mvc_reverse_arrays": {str(index): reverse_mapping[index] for index in sorted(reverse_mapping)},
            "completed_indices": completed, "unresolved_indices": [row["replicate_index"] for row in cohorts if row["status"] == "unknown"],
            "cohorts": cohorts, "diagnostic_counts": {"complete_menus": len(completed),
                "M0_only_maximal_diagnostics": sum(row["summary"]["maximal_eligible_models_diagnostic"] == ["M0"] for row in complete_rows),
                "multiple_incomparable_diagnostics": sum(row["summary"]["multiple_incomparable_eligible_models"] for row in complete_rows),
                "unknown_cohorts": 200 - len(completed)},
            **{field: "not_evaluated" for field in CLAIM_FIELDS}, "full_recovery_pass": None, "episode_terminal": False,
            "full_menu_wave_selected": False, "retry_selected": False,
            "interpretation": "Counts describe complete menus and their existing nested partial-order diagnostics only. Missing, technical and unadmitted cohorts remain unknown in fixed200; no false-selection rate, family arbitration, interval, early-kill or recovery gate verdict is computed.",
            "process_state_boundary": "The coordinator separately establishes selected terminal processes; this finite pool never polls or dispatches."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-indices", type=baseline.terminal_indices, required=True)
    parser.add_argument("--additional-mvc-array-job-id", type=baseline.array_id,
                        help="Explicitly admit completed null108/176 components from this selected MVC array")
    parser.add_argument("--mvc-repair-arrays", type=mvc_repairs.repair_arrays,
                        help="Explicit runtime1500 repair mapping, for example43:46599566,61:ARRAY,168:ARRAY")
    parser.add_argument("--mvc-reverse-arrays", type=mvc_reverses.reverse_arrays,
                        help="Explicit preserved-first/reverse1500 mapping, for example183:46600829,31:46601362")
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous summaries; choose a fresh output")
    result = pool(args.terminal_indices, json.loads(IMPLEMENTATION.read_text()),
                  args.additional_mvc_array_job_id, args.mvc_repair_arrays, args.mvc_reverse_arrays)
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({key: result[key] for key in ("phase", "completed_indices", "diagnostic_counts", "false_selection_rate")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
