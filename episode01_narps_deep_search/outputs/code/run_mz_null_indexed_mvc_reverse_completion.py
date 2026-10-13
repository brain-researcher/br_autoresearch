#!/usr/bin/env python3
"""Complete explicitly diagnosed indexed null MVC: preserve first1000, fit reverse1500.

All source reads, validation, replay, rescoring and fitting use the unchanged
indexed runner's bounded loop. No failed optimizer state or baseline is refit.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
from unittest.mock import patch

from qualify_assets import require_compute
import run_mz_null_indexed_mvc as indexed
import summarize_mz_null_indexed_menu as admission


MAX_SECONDS, RUNTIME_SWEEPS = 6600, 1500
IDENTITIES = {
    12: {"failed_job_id": "46592672", "failed_finished_utc": "2026-10-05T09:14:10.575469+00:00",
         "source_job_id": "46580411", "source_finished_utc": "2026-10-05T06:47:37.192916+00:00"},
    31: {"failed_job_id": "46592691", "failed_finished_utc": "2026-10-05T09:05:29.153076+00:00",
         "source_job_id": "46580477", "source_finished_utc": "2026-10-05T06:08:25.561996+00:00"},
    37: {"failed_job_id": "46592697", "failed_finished_utc": "2026-10-05T09:21:35.952939+00:00",
         "source_job_id": "46580479", "source_finished_utc": "2026-10-05T06:26:09.693122+00:00"},
    109: {"failed_job_id": "46592768", "failed_finished_utc": "2026-10-05T09:11:38.909699+00:00",
          "source_job_id": "46581542", "source_finished_utc": "2026-10-05T06:39:20.500623+00:00"},
    135: {"failed_job_id": "46592794", "failed_finished_utc": "2026-10-05T09:20:26.178459+00:00",
          "source_job_id": "46581744", "source_finished_utc": "2026-10-05T06:46:47.016312+00:00"},
    140: {"failed_job_id": "46592799", "failed_finished_utc": "2026-10-05T09:25:59.986947+00:00",
          "source_job_id": "46581745", "source_finished_utc": "2026-10-05T06:46:58.365639+00:00"},
    159: {"failed_job_id": "46592818", "failed_finished_utc": "2026-10-05T09:28:56.260092+00:00",
          "source_job_id": "46581750", "source_finished_utc": "2026-10-05T06:46:55.776760+00:00"},
    164: {"failed_job_id": "46592823", "failed_finished_utc": "2026-10-05T09:17:54.402056+00:00",
          "source_job_id": "46581751", "source_finished_utc": "2026-10-05T06:58:58.909195+00:00"},
    183: {"failed_job_id": "46593016", "failed_finished_utc": "2026-10-05T08:58:23.159202+00:00",
          "source_job_id": "46581756", "source_finished_utc": "2026-10-05T07:13:11.126436+00:00"},
    188: {"failed_job_id": "46593021", "failed_finished_utc": "2026-10-05T09:33:32.161845+00:00",
          "source_job_id": "46581757", "source_finished_utc": "2026-10-05T07:11:54.213374+00:00"},
}
QUALIFYING_REPORT = admission.RUNS / "20261005-MZ42-MVC-reverse-completion-46587905/mz_mvc_reverse_completion.json"
REPORT_NAME = "mz_null_indexed_mvc_reverse_completion.json"
PHASE = "M0_null_indexed_MVC_preserved1000_reverse1500_completion"
NONCONVERGENCE = "RuntimeError: Nonconverged fit is technical; it is not a recovery failure"


def failed_path(index):
    if type(index) is not int or index not in IDENTITIES:
        raise ValueError("Only explicitly identified selected null reverse completions are admitted")
    return admission.RUNS / f"20261005-M0-null-MVC-indexed-array-46592261/replicate{index}/mz_null_indexed_mvc.json"


def qualification_boundary(index):
    failed_path(index)
    # Preserve metadata already written by the first31/183 selected processes.
    if index in (31, 183):
        return "Coordinator separately selects each diagnosed null31/183 reverse completion; no pilot reread or wave selection"
    return "Coordinator separately selects each diagnosed indexed null reverse completion; no pilot reread or wave selection"


def repair_context(index, original, failed, source, people):
    """Bind the exact terminal failure and preserved fit to actual source people."""
    first_direction = {"training_runs": list(indexed.HALVES[0][0]), "scoring_runs": list(indexed.HALVES[0][1])}
    path = failed_path(index)
    identity = IDENTITIES[index]
    if (str(source.get("job_id")) != identity["source_job_id"] or source.get("finished_utc") != identity["source_finished_utc"]
            or failed.get("job_id") != identity["failed_job_id"] or failed.get("finished_utc") != identity["failed_finished_utc"]
            or failed.get("status") != "technical_rival_failure" or failed.get("error") != NONCONVERGENCE
            or failed.get("source_baseline_validated") is not True or failed.get("baseline_lineage") is not None
            or failed.get("internal_seconds") != 8400 or failed.get("scored_primary_directions") != [first_direction]
            or not isinstance(failed.get("fits"), list) or len(failed["fits"]) != 2):
        raise ValueError("Require this cohort's exact selected original1000 reverse failure and scored first fit")
    with patch.object(indexed, "RUNTIME_SWEEPS", 1000):
        admission.rival_identity(failed, "MVC", index, source, original, None)
        admission.menu.completed_support(failed)
        prior = admission.technical_rival(failed, "MVC", index, original, people)
        previous_runtime = indexed.runtime_specification(original)
    first, reverse = failed["fits"]
    history = reverse.get("objective_history")
    if (first.get("converged") is not True or reverse.get("converged") is not False
            or reverse.get("subject_optimizer_failures") != []
            or not isinstance(history, list) or len(history) != 1000
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in history)):
        raise ValueError("Require a validated first fit and exactly1000 finite failed reverse sweeps without subject failures")
    change = abs(history[-1] - history[-2])
    threshold = original["optimizer"]["joint_relative_tolerance"] * (1 + abs(history[-2]))
    if not math.isfinite(change) or not math.isfinite(threshold) or change <= threshold:
        raise ValueError("Reverse exhaustion must exceed the unchanged convergence threshold")
    metadata = {
        "kind": "second_half_1000_sweep_exhaustion_preserve_first_fresh_reverse1500",
        "failed_report": str(path), **identity,
        "failed_started_utc": failed["started_utc"], "failed_error": failed["error"],
        "original_maximum_sweeps": 500, "previous_runtime_maximum_sweeps": 1000,
        "preserved_first_validation_maximum_sweeps": 1000, "fresh_reverse_maximum_sweeps": RUNTIME_SWEEPS,
        "previous_runtime_implementation": previous_runtime,
        "previous_process_elapsed_seconds": prior["elapsed_seconds"],
        "previous_fit_elapsed_seconds": [fit["elapsed_seconds"] for fit in failed["fits"]],
        "previous_technical_attempt": prior, "first_half_original_fit_elapsed_seconds": first["elapsed_seconds"],
        "first_half_score_handling": "recomputed_from_preserved_parameters_no_saved_MVC_score_agreement_claim",
        "new_first_half_fit_elapsed_seconds_meaning": "copy_return_overhead_not_original_fit_cost",
        "final_reverse_objective_change": change, "unchanged_convergence_threshold": threshold,
        "nonconverged_checkpoint_state_reused": False,
        "second_half_initialization": "unchanged_fresh_prior_centers_and_recorded_first_sweep_offsets",
        "tolerances_priors_likelihood_support_and_first_convergence_rule_changed": False,
        "qualifying_runtime1500_report": str(QUALIFYING_REPORT),
        "cost_accounting": "prior_failed_MVC_process_plus_new_MVC_process_once; no_duplicate_first_fit_or_shared_baseline_cost",
    }
    return deepcopy(first), metadata


def run(args):
    require_compute()
    path = failed_path(args.replicate_index)
    if type(args.max_seconds) is not int or not 1 <= args.max_seconds <= MAX_SECONDS:
        raise ValueError("Keep the whole-process cap within1 through6600 seconds")
    output = args.output_dir.resolve()
    if any(output == failed_path(index).parent.resolve() or failed_path(index).parent.resolve() in output.parents for index in IDENTITIES):
        raise FileExistsError("Preserve all selected original failed attempts in place")
    original_source, original_fit = indexed.validate_source, indexed.fit_partially_pooled
    original_validate, original_score, original_write = indexed.validate_fit, indexed.null.shared.score_fit, indexed.write_report
    original = runtime = first = metadata = None
    calls, first_rescore_seconds = 0, None

    def validate_source(source, truth, people, index, spec):
        nonlocal original, runtime, first, metadata
        if index != args.replicate_index:
            raise ValueError("Only the selected fixed cohort is admitted")
        lineage = original_source(source, truth, people, index, spec)
        if lineage is not None:
            raise ValueError("Preserve this cohort's unchanged original500 baseline")
        original = deepcopy(spec)
        runtime = indexed.runtime_specification(original)
        first, metadata = repair_context(index, original, json.loads(path.read_text()), source, people)
        return lineage

    def fitter(training, model, bound, spec):
        nonlocal calls
        if model != "MVC" or bound != "fixed" or metadata is None or spec != runtime:
            raise ValueError("Only the validated preserved first and fresh reverse MVC are selected")
        calls += 1
        if calls == 1:
            return deepcopy(first)
        if calls == 2:
            return original_fit(training, model, bound, spec)
        raise ValueError("Only one preserved first fit and one fresh reverse fit are selected")

    def validate_fit(fit, model, train, held, spec, people):
        if model != "MVC" or metadata is None or spec != runtime:
            raise ValueError("Unexpected fit or runtime in the selected reverse completion")
        direction = (tuple(train), tuple(held))
        if direction == indexed.HALVES[0]:
            selected = metadata["previous_runtime_implementation"]
        elif direction == indexed.HALVES[1]:
            selected = runtime
        else:
            raise ValueError("Only the two original primary directions are admitted")
        return original_validate(fit, model, train, held, selected, people)

    def score_fit(fit, scoring, spec, check):
        nonlocal first_rescore_seconds
        began = indexed.time.perf_counter()
        values = original_score(fit, scoring, spec, check)
        if fit["model"] == "MVC" and tuple(fit["training_runs"]) == indexed.HALVES[0][0]:
            first_rescore_seconds = indexed.time.perf_counter() - began
        return values

    def write_checkpoint(directory, report):
        report.update(phase=PHASE, failed_report=str(path), repair_context_validated=metadata is not None,
            numerical_repair=deepcopy(metadata), original_maximum_sweeps=500, previous_runtime_maximum_sweeps=1000,
            preserved_first_validation_maximum_sweeps=1000, fresh_reverse_maximum_sweeps=RUNTIME_SWEEPS,
            wave_selected=False, wave_scope="single_indexed_MVC_reverse_completion", component_scope="MVC_only",
            retry_wave_selected=False, full_menu_wave_selected=False, retry_selected=False, automatic_retry=False,
            first_half_fit_reused=calls >= 1, fresh_second_half_fit_called=calls >= 2,
            first_half_rescored=first_rescore_seconds is not None, first_half_rescore_elapsed_seconds=first_rescore_seconds,
            first_half_copy_elapsed_seconds=report["fits"][0]["elapsed_seconds"] if report["fits"] else None,
            per_half_provenance=[
                {"training_runs": list(indexed.HALVES[0][0]), "scoring_runs": list(indexed.HALVES[0][1]),
                 "fit_source": str(path), "action": "reuse_validated_fit_and_rescore", "validation_maximum_sweeps": 1000},
                {"training_runs": list(indexed.HALVES[1][0]), "scoring_runs": list(indexed.HALVES[1][1]),
                 "fit_source": "current_completion", "action": "fresh_fit", "validation_maximum_sweeps": RUNTIME_SWEEPS}],
            interpretation=f"Only null{args.replicate_index}'s diagnosed reverse1000 MVC exhaustion is completed. Preserve and rescore the validated first1000 fit; freshly fit reverse1500. Original500 M0/MZ scores are replayed and retained. No family, rate, interval, early-kill, full-menu or gate verdict.",
            cost_accounting="elapsed_seconds is this completion; cumulative adds previous MVC process once, without adding original first-fit or shared baseline cost again")
        report["numerical_override"].update(kind="second_half_maximum_sweeps_only",
            previous_runtime_maximum_sweeps=1000, preserved_first_validation_maximum_sweeps=1000,
            fresh_fit_directions=[{"training_runs": list(indexed.HALVES[1][0]), "scoring_runs": list(indexed.HALVES[1][1])}],
            qualifying_runtime1500_report=str(QUALIFYING_REPORT),
            qualification_boundary=qualification_boundary(args.replicate_index))
        if metadata is not None and "elapsed_seconds" in report:
            report["cumulative_process_elapsed_seconds"] = metadata["previous_process_elapsed_seconds"] + report["elapsed_seconds"]
        original_write(directory, report)

    with ExitStack() as stack:
        for name, value in {"RUNTIME_SWEEPS": RUNTIME_SWEEPS, "MAX_SECONDS": MAX_SECONDS, "REPORT_NAME": REPORT_NAME,
                            "validate_source": validate_source, "fit_partially_pooled": fitter,
                            "validate_fit": validate_fit, "write_report": write_checkpoint}.items():
            stack.enter_context(patch.object(indexed, name, value))
        stack.enter_context(patch.object(indexed.null.shared, "score_fit", score_fit))
        return indexed.run(argparse.Namespace(replicate_index=args.replicate_index, implementation=args.implementation,
                                             output_dir=output, max_seconds=args.max_seconds))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicate-index", type=int, choices=tuple(IDENTITIES), required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    report = run(parser.parse_args(argv))
    print(json.dumps({key: report[key] for key in ("phase", "status", "rival_diagnostics_complete",
        "first_half_fit_reused", "fresh_second_half_fit_called", "elapsed_seconds", "cumulative_process_elapsed_seconds", "error")
        if key in report}, indent=2, allow_nan=False), flush=True)
    return 0 if report["rival_diagnostics_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
