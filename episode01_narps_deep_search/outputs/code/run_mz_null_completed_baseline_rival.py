#!/usr/bin/env python3
"""Fit MC, MV or MVZ on null35 only after its exact baseline completion46586603.

The simulation stays in the failed original worker's directory; the completed
four-fit baseline and scores stay in their distinct completion report. Adapt
only the existing null runner's source identity/validation and checkpoints.
The coordinator selects execution after terminal predecessor/component checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_baseline_completion as completion
import run_mz_null_indexed_simple_rival as indexed_simple
import run_mz_null_rival as null


MODELS = ("MC", "MV", "MVZ")
MAX_SECONDS = {"MC": 2400, "MV": 2400, "MVZ": 3300}
COMPLETION_JOB_ID = "46586603"
COMPLETION_DIRECTORY = completion.baseline.RUNS / f"20261005-M0-null-baseline35-completion-{COMPLETION_JOB_ID}"
COMPLETION_REPORT = COMPLETION_DIRECTORY / completion.REPORT_NAME
FAILED_REPORT = completion.SOURCE / completion.SOURCE_REPORT
REPORT_NAME = "mz_null_completed_baseline_rival.json"
PHASE = "M0_null_completed_baseline35_rival_diagnostics"
PRESERVED_DIRECTORIES = (*indexed_simple.PRESERVED_DIRECTORIES, COMPLETION_DIRECTORY,
                         completion.DESIGN_SOURCE.parent)


def validate_completed_baseline(report, failed, truth, people, spec):
    """Validate the new phase without relabeling it as an original success."""
    preserved, metadata = completion.completion_context(failed, truth, people, spec)
    if (not isinstance(report, dict) or report.get("analysis_id") != "behavioral_mechanism_kill_test"
            or report.get("phase") != completion.PHASE or str(report.get("job_id")) != COMPLETION_JOB_ID
            or report.get("generating_model") != "M0" or report.get("fitted_models") != ["M0", "MZ"]
            or type(report.get("replicate_index")) is not int or report["replicate_index"] != completion.REPLICATE_INDEX
            or type(report.get("seed")) is not int or report["seed"] != completion.SEED
            or report.get("SESOI_level") != 0.
            or report.get("focal_generating_truth") != completion.indexed.representative.NULL_TRUTH
            or report.get("implementation") != spec
            or report.get("source_dir") is None or Path(report["source_dir"]).resolve() != completion.SOURCE.resolve()
            or report.get("source_report") != completion.SOURCE_REPORT
            or str(report.get("source_job_id")) != completion.SOURCE_JOB_ID
            or report.get("generating_parameters") != str(completion.SOURCE / "generating_parameters.json")
            or report.get("simulated_events") != str(completion.SOURCE / "simulated_events.csv.gz")
            or any(report.get(key) != value for key, value in metadata.items())):
        raise ValueError("Require the exact null35 completion46586603 and original failed-source context")
    for field in ("started_utc", "finished_utc"):
        value = report.get(field)
        if not isinstance(value, str) or datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("The completed baseline requires dated terminal provenance")
    if (report.get("status") != "pairwise_diagnostic_completed" or report.get("pairwise_diagnostic_complete") is not True
            or report.get("source_context_validated") is not True or report.get("preserved_scores_recomputed") is not True
            or report.get("scored_primary_directions") !=
                [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in null.HALVES]
            or report.get("scored_fit_prefix") !=
                [{"model": model, "training_runs": list(train), "scoring_runs": list(held)}
                 for model, train, held in completion.EXPECTED]
            or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]):
        raise ValueError("Only the completed four-fit, two-direction baseline may be reused")
    if (any(report.get(field) is not False for field in
            ("simulation_regenerated", "simulation_copied", "saved_completed_fits_refitted", "unfinished_fit_state_reused",
             "observed_choices_or_RTs_used", "old_synthetic_responses_used", "wave_selected", "retry_wave_selected",
             "full_menu_wave_selected", "episode_terminal"))
            or report.get("selected_replicates") != [completion.REPLICATE_INDEX]
            or report.get("fixed_cohort_count") != 200
            or report.get("component_scope") != "single_M0_MZ_baseline_completion"
            or report.get("saved_score_agreement") != "not_evaluated_original_failure_has_no_saved_scores"
            or any(report.get(field) != "not_evaluated" for field in null.CLAIM_FIELDS)
            or report.get("early_kill", "not_evaluated") != "not_evaluated"
            or report.get("full_recovery_pass") is not None
            or "counts_as_curtailment_failure" in report or "core_family_selection_pass" in report):
        raise ValueError("Retain the single completion's diagnostic-only scope and untouched simulation")
    completion.fixed.fit_panel(report, completion.EXPECTED, spec, people)
    # This completion runner deliberately preserves elapsed_seconds too. There
    # is no timing-field exclusion: all three fit dictionaries must be exact.
    if report["fits"][:3] != preserved:
        raise ValueError("Every preserved fit parameter, history, diagnostic and original duration must match")
    scores = completion.fixed.score_panel(report, ("M0", "MZ"))
    if set(scores) != set(people) or any(scores[person]["version"] != people[person][0]["task_version"] for person in people):
        raise ValueError("Completed baseline score participant/task-version identities differ")
    if report.get("nested_comparison") != completion.fixed.nested_contrast(scores, "MZ", "M0"):
        raise ValueError("The complete baseline nested diagnostic must match its recorded scores")
    elapsed = completion.duration(report.get("elapsed_seconds"), "Baseline completion process cost")
    cumulative = completion.duration(report.get("cumulative_process_elapsed_seconds"), "Cumulative baseline process cost")
    for field in ("source_validation_and_copy_seconds", "preserved_fit_rescore_seconds", "fresh_fit_score_seconds"):
        completion.duration(report.get(field), field)
    fresh = completion.duration(report.get("fresh_fit_elapsed_seconds"), "Fresh reverse MZ fit cost")
    if (type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= 2700
            or fresh != report["fits"][3].get("elapsed_seconds") or fresh > elapsed
            or cumulative != metadata["previous_process_elapsed_seconds"] + elapsed):
        raise ValueError("Keep original/completion costs distinct and count each process once")
    return {
        "original_failed_report": str(FAILED_REPORT), "original_worker_job_id": completion.SOURCE_JOB_ID,
        "original_finished_utc": failed["finished_utc"],
        "original_process_elapsed_seconds": metadata["previous_process_elapsed_seconds"],
        "completion_report": str(COMPLETION_REPORT), "completion_job_id": COMPLETION_JOB_ID,
        "completion_finished_utc": report["finished_utc"], "completion_process_elapsed_seconds": elapsed,
        "cumulative_baseline_process_elapsed_seconds": cumulative,
        "per_fit_provenance": metadata["per_fit_provenance"],
    }


def run(args):
    require_compute()
    if args.model not in MODELS:
        raise ValueError("Select only MC, MV or MVZ on completed null35")
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS[args.model]:
        raise ValueError("Keep the selected MC/MV2400s or MVZ3300s whole-process bound")
    output = args.output_dir.resolve()
    if any(output == path.resolve() or path.resolve() in output.parents for path in PRESERVED_DIRECTORIES):
        raise FileExistsError("Preserve the original baseline arrays, completion and null0 components")
    original_writer = null.write_report
    lineage = None

    def validate_source(report, truth, people, spec):
        nonlocal lineage
        # The unchanged runner has already loaded the exact completion report
        # and original truth/cohort inside its whole-process timer.
        failed = json.loads(FAILED_REPORT.read_text())
        lineage = validate_completed_baseline(report, failed, truth, people, spec)

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, source_job_id=COMPLETION_JOB_ID if lineage is not None else None,
                      source_baseline_validated=lineage is not None, baseline_lineage=lineage,
                      selected_replicates=[completion.REPLICATE_INDEX], selected_models=[args.model],
                      fixed_cohort_count=200, component_scope=f"{args.model}_only",
                      wave_selected=False, retry_wave_selected=False, full_menu_wave_selected=False,
                      simulation_copied=False,
                      interpretation=f"One {args.model} component on null35 after the exact baseline completion. The original simulation remains in place, completed M0/MZ scores are replayed and retained, and only this rival is fitted in both halves under original500. No full-menu selection, rate, interval, gate or early-kill verdict is supplied.",
                      cost_accounting="elapsed_seconds is this rival process only. baseline_lineage references shared prior costs; those costs are not added to each rival's measured process cost.",
                      process_state_boundary="The coordinator separately checks terminal completion and earlier component jobs before selecting this adapter; report fields are not scheduler evidence.")
        original_writer(directory, report)

    runtime_args = argparse.Namespace(**{**vars(args), "source_dir": completion.SOURCE})
    # An absolute SOURCE_REPORT makes SOURCE / SOURCE_REPORT select the new
    # report while all truth/event reads remain at the original SOURCE. No
    # source is copied or relabeled, and all bindings are restored afterward.
    with patch.object(null, "SOURCE", completion.SOURCE), \
            patch.object(null, "SOURCE_REPORT", str(COMPLETION_REPORT)), \
            patch.object(null, "SOURCE_JOB_ID", None), \
            patch.object(null, "SEED", completion.SEED), \
            patch.object(null, "REPLICATE_INDEX", completion.REPLICATE_INDEX), \
            patch.object(null, "REPORT_NAME", REPORT_NAME), \
            patch.object(null, "validate_source", validate_source), \
            patch.object(null, "write_report", write_checkpoint):
        return null.run(runtime_args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "fitted_model", "replicate_index", "seed",
          "source_job_id", "source_baseline_validated", "rival_diagnostics_complete", "diagnostics", "elapsed_seconds", "error")
          if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
