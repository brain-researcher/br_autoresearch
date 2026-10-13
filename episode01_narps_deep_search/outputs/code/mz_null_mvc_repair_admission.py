"""Optional null43/61/168 MVC repair admission for the finite menu pool.

Import the producer lazily: it imports the existing pool for prior1000 report
validation. This helper does not change those producer-used validation paths.
The caller first validates the original500 baseline and actual saved cohort.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
from unittest.mock import patch


INDICES = frozenset((43, 61, 168))
RUNS = Path(__file__).resolve().parents[2] / "outputs/runs/behavioral_mechanism_kill_test"


def validate_mapping(mapping):
    if mapping is None:
        return {}
    if (not isinstance(mapping, dict)
            or any(type(index) is not int or index not in INDICES
                   or type(array) is not int or array < 1 for index, array in mapping.items())):
        raise ValueError("Map only null43/61/168 to explicit positive repair array IDs")
    return dict(mapping)


def repair_arrays(value):
    mapping = {}
    for token in value.split(","):
        pair = [part.strip() for part in token.split(":")]
        if len(pair) != 2 or any(not part.isdecimal() for part in pair):
            raise argparse.ArgumentTypeError("Use explicit index:array pairs, for example43:46599566")
        index, array = map(int, pair)
        if index in mapping:
            raise argparse.ArgumentTypeError("Each repair index must appear only once")
        mapping[index] = array
    try:
        return validate_mapping(mapping)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def input_paths(index, mapping):
    selected = validate_mapping(mapping)
    if index not in selected:
        return None
    return {
        "MVC": RUNS / f"20261005-M0-null-MVC-repair-array-{selected[index]}" /
            f"replicate{index}/mz_null_indexed_mvc_repair.json",
        "failed_MVC": RUNS / f"20261005-M0-null-MVC-indexed-array-46592261" /
            f"replicate{index}/mz_null_indexed_mvc.json",
    }


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Retain finite nonnegative original and repaired MVC process costs")
    return value


def validate_repair(report, failed, base, original, baseline, people, index):
    """Return completed scores or technical evidence; missing prior lineage returnsNone."""
    # Deliberately local. The producer depends on the old pool validators, so
    # importing it during pool-module initialization would create a cycle.
    import run_mz_null_indexed_mvc_repair as producer

    admission = producer.admission
    path = producer.failed_path(index)
    directory, source_report = producer.indexed.source_paths(index)
    technical = report.get("status") == "technical_rival_failure"
    job = str(report.get("job_id", ""))
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != producer.PHASE
            or report.get("generating_model") != "M0" or report.get("fitted_model") != "MVC"
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != index
            or type(report.get("seed")) is not int or report["seed"] != 1300037 + 10007 * index
            or report.get("source_dir") != str(directory) or report.get("source_report") != str(source_report)
            or report.get("failed_report") != str(path) or not job.isdecimal() or int(job) < 1):
        raise ValueError("Require the selected actual null repair phase, model, index and source identity")
    if (any(report.get(field) != "not_evaluated" for field in producer.indexed.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_M0_MZ_refitted", "observed_choices_or_RTs_used", "old_synthetic_responses_used",
                "wave_selected", "retry_selected", "retry_wave_selected", "automatic_retry", "full_menu_wave_selected"))
            or report.get("selected_replicates") != [index] or report.get("selected_models") != ["MVC"]
            or report.get("fixed_cohort_count") != 200 or report.get("component_scope") != "MVC_only"
            or report.get("wave_scope") != "single_indexed_MVC_numerical_repair"
            or report.get("original_maximum_sweeps") != 500 or report.get("previous_runtime_maximum_sweeps") != 1000
            or report.get("fitted_maximum_sweeps") != 1500
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= producer.MAX_SECONDS):
        raise ValueError("Preserve the bounded single-component original500/previous1000/fresh1500 scope")
    terminal = admission.terminal_metadata(report)
    runtime = deepcopy(original)
    runtime["optimizer"]["maximum_sweeps"] = 1500
    for field, expected in (("implementation", original), ("runtime_implementation", runtime)):
        if (not technical or field in report) and report.get(field) != expected:
            raise ValueError("Only the fresh MVC maximum-sweeps ceiling may change")
    override = admission.expected_override()
    override.update(runtime_maximum_sweeps=1500, qualifying_runtime1500_report=str(producer.QUALIFYING_REPORT),
        qualification_boundary="Coordinator records runtime1500 first-convergence qualification and separately selects this diagnosed cohort; this worker never rereads pilots or selects a wave")
    if report.get("numerical_override") != override:
        raise ValueError("Preserve the exact fresh-both runtime override and qualification provenance")
    if technical:
        if (report.get("rival_diagnostics_complete") is not False or not isinstance(report.get("error"), str)
                or not report["error"] or "participant_scores" in report
                or report.get("completed_rivals") != [] or report.get("diagnostics") != {}):
            raise ValueError("A technical repair supplies no completed score panel")
    elif (report.get("status") != "rival_diagnostics_completed" or report.get("rival_diagnostics_complete") is not True
            or report.get("completed_rivals") != ["MVC"]
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in producer.indexed.HALVES]):
        raise ValueError("Require both completed and scored fresh MVC directions")
    validated = report.get("repair_context_validated")
    if validated is False:
        if (not technical or report.get("numerical_repair") is not None
                or report.get("source_baseline_validated") is not False or report.get("source_job_id") is not None
                or report.get("saved_scores_replayed") is not False or report.get("fits") != []
                or report.get("scored_primary_directions") != [] or "cumulative_process_elapsed_seconds" in report):
            raise ValueError("An unvalidated technical repair cannot claim source, replay, fits or prior costs")
        return {**terminal, "status": report["status"], "error": report["error"], "returned_fits": []}
    if validated is not True or not isinstance(report.get("numerical_repair"), dict):
        raise ValueError("Require explicit repair validation and recorded original-failure context")
    if failed is None:
        return None
    expected = producer.validate_failure(index, failed, base, people, original)
    if report["numerical_repair"] != expected:
        raise ValueError("Preserve the exact selected prior1000 failure, original specification and consumed costs")
    source_validated = report.get("source_baseline_validated")
    if source_validated is True:
        if str(report.get("source_job_id")) != str(base["job_id"]) or report.get("source_finished_utc") != base["finished_utc"]:
            raise ValueError("The repair must bind to this exact validated baseline completion")
        admission.menu.completed_support(report)
    elif (not technical or source_validated is not False or report.get("source_job_id") is not None
            or report.get("fits") != [] or report.get("scored_primary_directions") != []
            or report.get("saved_scores_replayed") is not False):
        raise ValueError("An unvalidated source cannot supply fitted or replayed scores")
    if duration(report.get("cumulative_process_elapsed_seconds")) != expected["previous_process_elapsed_seconds"] + report["elapsed_seconds"]:
        raise ValueError("Count previous failed MVC plus this MVC process once, excluding shared baseline costs")
    if technical:
        # Existing technical-prefix checks validate converged returned fits at
        # the new ceiling. The immutable prior report was checked at1000 above.
        with patch.object(producer.indexed, "RUNTIME_SWEEPS", 1500):
            evidence = admission.technical_rival(report, "MVC", index, original, people)
        return {**evidence, "numerical_repair": report["numerical_repair"],
                "cumulative_process_elapsed_seconds": report["cumulative_process_elapsed_seconds"]}
    admission.validate_replay(report)
    fits = [("MVC", train, held) for train, held in producer.indexed.HALVES]
    admission.menu.fixed.fit_panel(report, fits, runtime, people)
    scores = admission.menu.fixed.score_panel(report, ("M0", "MZ", "MVC"))
    if set(scores) != set(baseline) or any(scores[person][key] != baseline[person][key]
                                         for person in baseline for key in ("M0", "MZ", "version")):
        raise ValueError("Preserve baseline participant/task-version identities and every saved score last bit")
    return scores
