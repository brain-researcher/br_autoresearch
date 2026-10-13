"""Optional exact identified indexed null reverse-completion admission.

The caller validates the actual original500 baseline and cohort first. The
producer is imported only inside validation to avoid its existing pool import
forming an initialization cycle. This helper computes no scientific contrasts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mz_null_mvc_repair_admission import duration


INDICES = frozenset((12, 31, 37, 109, 135, 140, 159, 164, 183, 188))
RUNS = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test"


def validate_mapping(mapping):
    if mapping is None:
        return {}
    if (not isinstance(mapping, dict) or any(type(index) is not int or index not in INDICES
            or type(array) is not int or array < 1 for index, array in mapping.items())):
        raise ValueError("Map only explicitly identified null reverse completions to positive array IDs")
    return dict(mapping)


def reverse_arrays(value):
    mapping = {}
    for token in value.split(","):
        pair = [part.strip() for part in token.split(":")]
        if len(pair) != 2 or any(not part.isdecimal() for part in pair):
            raise argparse.ArgumentTypeError("Use explicit index:array pairs, for example183:70000000")
        index, array = map(int, pair)
        if index in mapping:
            raise argparse.ArgumentTypeError("Each reverse-completion index must appear only once")
        mapping[index] = array
    try:
        return validate_mapping(mapping)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def input_paths(index, mapping):
    selected = validate_mapping(mapping)
    if index not in selected:
        return None
    return {"MVC": RUNS / f"20261005-M0-null-MVC-reverse-completion-array-{selected[index]}" /
                    f"replicate{index}/mz_null_indexed_mvc_reverse_completion.json",
            "failed_MVC": RUNS / f"20261005-M0-null-MVC-indexed-array-46592261/replicate{index}/mz_null_indexed_mvc.json"}


def validate_completion(report, failed, base, original, baseline, people, index):
    """Return validated scores, technical evidence, orNone for missing prior lineage."""
    import run_mz_null_indexed_mvc_reverse_completion as producer

    admission = producer.admission
    path = producer.failed_path(index)
    directory, source_report = producer.indexed.source_paths(index)
    halves = producer.indexed.HALVES
    directions = [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in halves]
    technical = report.get("status") == "technical_rival_failure"
    job = str(report.get("job_id", ""))
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != producer.PHASE
            or report.get("generating_model") != "M0" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != 1300037 + 10007 * index
            or report.get("source_dir") != str(directory) or report.get("source_report") != str(source_report)
            or report.get("failed_report") != str(path) or not job.isdecimal() or int(job) < 1):
        raise ValueError("Require the selected actual reverse-completion phase, cohort, seed and source")
    if (any(report.get(field) != "not_evaluated" for field in producer.indexed.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_M0_MZ_refitted", "observed_choices_or_RTs_used", "old_synthetic_responses_used",
                "wave_selected", "retry_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected"))
            or report.get("selected_replicates") != [index] or report.get("selected_models") != ["MVC"]
            or report.get("fixed_cohort_count") != 200 or report.get("component_scope") != "MVC_only"
            or report.get("wave_scope") != "single_indexed_MVC_reverse_completion"
            or report.get("original_maximum_sweeps") != 500 or report.get("previous_runtime_maximum_sweeps") != 1000
            or report.get("preserved_first_validation_maximum_sweeps") != 1000 or report.get("fresh_reverse_maximum_sweeps") != 1500
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= producer.MAX_SECONDS
            or report.get("baseline_lineage") is not None):
        raise ValueError("Preserve the bounded single-component original500/first1000/reverse1500 scope")
    terminal = admission.terminal_metadata(report)
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = 1500
    for field, expected in (("implementation", original), ("runtime_implementation", runtime)):
        if (not technical or field in report) and report.get(field) != expected:
            raise ValueError("Only the fresh reverse MVC maximum-sweeps ceiling may change")
    override = admission.expected_override()
    override.update(kind="second_half_maximum_sweeps_only", runtime_maximum_sweeps=1500,
        previous_runtime_maximum_sweeps=1000, preserved_first_validation_maximum_sweeps=1000,
        fresh_fit_directions=[directions[1]], qualifying_runtime1500_report=str(producer.QUALIFYING_REPORT),
        qualification_boundary=producer.qualification_boundary(index))
    provenance = [{**directions[0], "fit_source": str(path), "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 1000},
                  {**directions[1], "fit_source": "current_completion", "action": "fresh_fit", "validation_maximum_sweeps": 1500}]
    if report.get("numerical_override") != override or report.get("per_half_provenance") != provenance:
        raise ValueError("Retain exact preserved-first and fresh-reverse numerical provenance")
    fits = report.get("fits")
    reused, called, rescored = (report.get(field) for field in
                               ("first_half_fit_reused", "fresh_second_half_fit_called", "first_half_rescored"))
    if (not isinstance(fits, list) or len(fits) > 2 or any(not isinstance(fit, dict) for fit in fits)
            or any(type(value) is not bool for value in (reused, called, rescored))
            or (fits and not reused) or (len(fits) == 2 and not called)
            or (rescored and (not reused or not fits))
            or (called and (not reused or not rescored or not report.get("scored_primary_directions")))):
        raise ValueError("Returned/scored prefixes must agree with the preserved and fresh invocation metadata")
    if fits:
        if duration(report.get("first_half_copy_elapsed_seconds")) != duration(fits[0].get("elapsed_seconds")):
            raise ValueError("The new first-fit elapsed field is exactly copy-return overhead")
    elif report.get("first_half_copy_elapsed_seconds") is not None:
        raise ValueError("No returned first fit can claim completed copy overhead")
    if rescored:
        duration(report.get("first_half_rescore_elapsed_seconds"))
    elif report.get("first_half_rescore_elapsed_seconds") is not None:
        raise ValueError("Unfinished first-fit rescoring cannot claim completed rescore overhead")
    if technical:
        if (report.get("rival_diagnostics_complete") is not False or not isinstance(report.get("error"), str)
                or not report["error"] or "participant_scores" in report
                or report.get("completed_rivals") != [] or report.get("diagnostics") != {}):
            raise ValueError("A technical completion supplies no successful score panel")
    elif (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or report.get("completed_rivals") != ["MVC"] or report.get("scored_primary_directions") != directions
            or len(fits) != 2 or not (reused and called and rescored)):
        raise ValueError("Require the rescored preserved first and completed fresh reverse")
    if report.get("repair_context_validated") is False:
        if (not technical or report.get("numerical_repair") is not None
                or report.get("source_baseline_validated") is not False or report.get("source_job_id") is not None
                or report.get("saved_scores_replayed") is not False or fits != []
                or report.get("scored_primary_directions") != [] or reused or called or rescored
                or "cumulative_process_elapsed_seconds" in report):
            raise ValueError("An unvalidated technical completion cannot claim source, replay, reuse or prior cost")
        return {**terminal, "status": report["status"], "error": report["error"], "returned_fits": []}
    if report.get("repair_context_validated") is not True or not isinstance(report.get("numerical_repair"), dict):
        raise ValueError("Require explicit exact prior-failure context")
    if failed is None:
        return None
    first, expected = producer.repair_context(index, original, failed, base, people)
    if report["numerical_repair"] != expected:
        raise ValueError("Preserve exact prior1000 failure, validated first fit and consumed costs")
    if report.get("source_baseline_validated") is True:
        if str(report.get("source_job_id")) != str(base["job_id"]) or report.get("source_finished_utc") != base["finished_utc"]:
            raise ValueError("Bind the completion to the exact validated baseline worker and finish")
        admission.menu.completed_support(report)
    elif (not technical or report.get("source_baseline_validated") is not False or report.get("source_job_id") is not None
            or fits != [] or report.get("scored_primary_directions") != [] or report.get("saved_scores_replayed") is not False):
        raise ValueError("An unvalidated source cannot supply fits or scores")
    if duration(report.get("cumulative_process_elapsed_seconds")) != expected["previous_process_elapsed_seconds"] + report["elapsed_seconds"]:
        raise ValueError("Count the previous MVC process and new process once without duplicate first-fit costs")
    if fits and ({key: value for key, value in fits[0].items() if key != "elapsed_seconds"}
                 != {key: value for key, value in first.items() if key != "elapsed_seconds"}):
        raise ValueError("Retain every preserved first-fit parameter, history and diagnostic except copied elapsed")
    previous_runtime = expected["previous_runtime_implementation"]
    if technical:
        old_fit_spec = admission.fit_spec

        def fit_spec(spec, selected_index, model, position=0):
            if spec != original or selected_index != index or model != "MVC" or position not in (0, 1):
                return old_fit_spec(spec, selected_index, model, position)
            return previous_runtime if position == 0 else runtime

        with patch.object(admission, "fit_spec", fit_spec):
            evidence = admission.technical_rival(report, "MVC", index, original, people)
        return {**evidence, "numerical_repair": report["numerical_repair"], "per_half_provenance": provenance,
                "first_half_copy_elapsed_seconds": report["first_half_copy_elapsed_seconds"],
                "first_half_rescore_elapsed_seconds": report["first_half_rescore_elapsed_seconds"],
                "cumulative_process_elapsed_seconds": report["cumulative_process_elapsed_seconds"]}
    admission.validate_replay(report)
    for fit, (train, held), spec in zip(fits, halves, (previous_runtime, runtime)):
        admission.menu.fixed.validate_fit(fit, "MVC", train, held, spec, people)
    scores = admission.menu.fixed.score_panel(report, ("M0", "MZ", "MVC"))
    if set(scores) != set(baseline) or any(scores[person][key] != baseline[person][key]
                                         for person in baseline for key in ("M0", "MZ", "version")):
        raise ValueError("Preserve all original500 baseline score last bits and participant/task-version identities")
    return scores
