#!/usr/bin/env python3
"""Pool four explicit MZ0 LOO reports within participant, never as replications.

The coordinator establishes terminal processes. Only these four reports and
the original implementation are read; their qualified source is not reread.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_leave_one_run_out as loo


PHASE = "MZ0_four_leave_one_run_out_nested_diagnostics"
HELD_RUNS = (1, 2, 3, 4)
NORMALIZATION = "sum_of64_heldout_trial_log_likelihoods_divided_by64_per_participant"


def parent_job(value):
    try:
        job = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Supply the explicit positive parent job ID") from error
    if type(value) is bool or job < 1 or str(job) != str(value):
        raise argparse.ArgumentTypeError("Supply the explicit positive parent job ID")
    return job


def report_paths(source_root, job_id):
    job_id = parent_job(job_id)
    root = Path(source_root).resolve()
    expected = loo.SOURCE.parent / f"20261005-MZ-LOO-{job_id}"
    if root != expected.resolve():
        raise ValueError("Use the explicit selected parent-job LOO directory")
    return {held: root / f"heldout{held}" / loo.REPORT_NAME for held in HELD_RUNS}


def finite_duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Retain finite nonnegative process/fit durations")
    return value


def validate_report(report, held, job_id, implementation):
    """Return descriptive row, completed scores, primary panel and source identity."""
    train = [run for run in HELD_RUNS if run != held]
    expected_fits = [{"model": model, "training_runs": train, "scoring_runs": [held]} for model in ("M0", "MZ")]
    technical = report.get("status") == "technical_loo_failure"
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != loo.PHASE
            or report.get("generating_model") != "MZ" or type(report.get("replicate_index")) is not int or report["replicate_index"] != 0
            or report.get("seed") != loo.shared.SEED or str(report.get("job_id")) != str(job_id)
            or type(report.get("held_out_run")) is not int or report["held_out_run"] != held
            or report.get("training_runs") != train or report.get("scoring_runs") != [held]
            or report.get("fitted_models") != ["M0", "MZ"]
            or report.get("source_dir") != str(loo.SOURCE)
            or report.get("source_report") != str(loo.SOURCE / loo.SOURCE_REPORT)
            or report.get("score_normalization") != NORMALIZATION):
        raise ValueError("Each report must retain its selected parent, MZ0 source and unique held-out run")
    if (any(report.get(field) != "not_evaluated" for field in loo.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("selected_replicates") != [0]
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_LOO", "observed_choices_or_RTs_used", "wave_selected",
                "automatic_retry", "LOO_runs_are_independent_replications"))
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= loo.MAX_SECONDS):
        raise ValueError("Preserve the finite diagnostic-only sensitivity scope")
    if (not technical or "implementation" in report) and report.get("implementation") != implementation:
        raise ValueError("The LOO fits must use the unchanged original500 implementation")
    elapsed = finite_duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.tzinfo is None or finished.tzinfo is None or finished < began:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each supplied report must be a terminal recorded attempt") from error
    row = {"held_out_run": held, "training_runs": train, "job_id": str(job_id),
           "elapsed_seconds": elapsed, "finished_utc": report["finished_utc"]}
    fits, scored = report.get("fits"), report.get("scored_fits")
    if (not isinstance(fits, list) or len(fits) > 2 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored) + 1
            or scored != expected_fits[:len(scored)]):
        raise ValueError("Retain the ordered returned/scored M0/MZ prefix")
    if technical:
        if (report.get("loo_diagnostic_complete") is not False or not isinstance(report.get("error"), str)
                or not report["error"] or "participant_scores" in report or "nested_diagnostic" in report):
            raise ValueError("A technical LOO attempt cannot supply completed scores or a diagnostic")
        row.update(status="technical_unknown", error=report["error"])
    elif (report.get("status") != "loo_diagnostic_completed" or report.get("loo_diagnostic_complete") is not True
            or len(fits) != 2 or scored != expected_fits):
        raise ValueError("A completed LOO report requires both fitted and scored models")
    else:
        row["status"] = "complete_diagnostic"
    validated, replayed = report.get("source_baseline_validated"), report.get("saved_primary_scores_replayed")
    if type(validated) is not bool or type(replayed) is not bool:
        raise ValueError("Source validation and primary replay must be explicit")
    source_identity = None
    if validated:
        source_job = str(report.get("source_job_id", ""))
        source_finish = report.get("source_finished_utc")
        if (not source_job.isdecimal() or int(source_job) < 1 or not isinstance(source_finish, str) or not source_finish
                or [report.get(key) for key in ("participants", "runs", "trials")] != [108, 432, 27648]
                or report.get("training_trials_per_participant") != 192 or report.get("heldout_trials_per_participant") != 64):
            raise ValueError("Validated sources require the same full108-person192/64 split support")
        source_identity = (source_job, source_finish)
    elif (not technical or replayed or fits or report.get("source_job_id") is not None
            or report.get("source_finished_utc") is not None):
        raise ValueError("An unvalidated source cannot supply fitted or replayed evidence")
    primary = None
    if replayed:
        discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_score_replay_absolute_tolerance") != loo.shared.REPLAY_ABSOLUTE_TOLERANCE
                or type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
                or not 0 <= discrepancy <= loo.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("Preserve the original primary-score replay at1e-12")
        primary = loo.fixed.score_panel({"participant_scores": report.get("saved_primary_participant_scores")}, ("M0", "MZ"))
    elif not technical or fits or "saved_primary_participant_scores" in report:
        raise ValueError("New fitted scores require successful original primary replay")
    returned = []
    for position, (fit, context) in enumerate(zip(fits, expected_fits)):
        if (any(fit.get(key) != value for key, value in context.items()) or fit.get("rival") != "fixed"
                or fit.get("participants") != sorted(primary)):
            raise ValueError("Every returned fit must retain its model, actual participants and192/64 context")
        duration = finite_duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            loo.validate_fit(fit, context["model"], train, [held], implementation, primary)
        elif fit.get("converged") is not False or position < len(scored) or not technical:
            raise ValueError("Only converged validated fits can supply a scored model")
        returned.append({**context, "converged": fit["converged"], "elapsed_seconds": duration})
    if technical:
        row.update(returned_fits=returned, scored_fits=scored)
        return row, None, primary, source_identity
    scores = loo.fixed.score_panel(report, ("M0", "MZ"))
    if set(scores) != set(primary) or any(scores[person]["version"] != primary[person]["version"] for person in scores):
        raise ValueError("LOO scores must retain the qualified source participants and task versions")
    diagnostic = loo.fixed.nested_contrast(scores, "MZ", "M0")
    if report.get("nested_diagnostic") != diagnostic:
        raise ValueError("The recorded per-run diagnostic differs from the unchanged nested calculation")
    sign = lambda value: int(value > 0) - int(value < 0)
    row.update(nested_diagnostic=diagnostic, mean_advantage_sign=sign(diagnostic["mean_gain_nats_per_trial"]),
               version_advantage_signs={version: sign(value) for version, value in diagnostic["version_means"].items()})
    return row, scores, primary, source_identity


def pool(source_root, job_id, implementation):
    paths = report_paths(source_root, job_id)
    if implementation["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Preserve the original500 implementation")
    rows, panels, source_identity, primary = [], {}, None, None
    for held in HELD_RUNS:
        try:
            report = json.loads(paths[held].read_text())
        except FileNotFoundError:
            rows.append({"held_out_run": held, "status": "missing_unknown", "report": str(paths[held])})
            continue
        row, scores, saved_primary, identity = validate_report(report, held, job_id, implementation)
        row["report"] = str(paths[held])
        rows.append(row)
        if identity is not None:
            if source_identity is not None and identity != source_identity:
                raise ValueError("LOO reports must share the exact original source worker and completion")
            source_identity = identity
        if saved_primary is not None:
            if primary is not None and saved_primary != primary:
                raise ValueError("LOO reports must retain identical original primary-score last bits")
            primary = saved_primary
        if scores is not None:
            panels[held] = scores
    complete = len(panels) == 4
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ",
        "replicate_index": 0, "seed": loo.shared.SEED, "source_root": str(Path(source_root).resolve()),
        "job_id": str(job_id), "source_dir": str(loo.SOURCE), "source_report": str(loo.SOURCE / loo.SOURCE_REPORT),
        "source_job_id": source_identity[0] if source_identity else None,
        "source_finished_utc": source_identity[1] if source_identity else None,
        "implementation": implementation, "status": "four_run_diagnostic_completed" if complete else "incomplete_loo_diagnostics",
        "pooled_diagnostic_complete": complete, "complete_held_out_runs": sorted(panels),
        "unknown_held_out_runs": [held for held in HELD_RUNS if held not in panels], "per_run": rows,
        "selected_replicates": [0], "wave_selected": False, "automatic_retry": False,
        "LOO_runs_are_independent_replications": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in loo.CLAIM_FIELDS},
        "interpretation": "Equal averaging occurs within each of108 participants across all four64-trial held-out runs. Overlapping training folds are sensitivities, not independent replications. Missing or technical folds leave the pooled diagnostic unknown. Positive-run and EI/ER signs are descriptive contract diagnostics, not a family, rate, interval, curtailment or gate verdict."}
    if complete:
        scores = {person: {"version": primary[person]["version"],
                           **{model: sum(panels[held][person][model] for held in HELD_RUNS) / 4
                              for model in ("M0", "MZ")}} for person in sorted(primary)}
        loo.fixed.score_panel({"participant_scores": scores}, ("M0", "MZ"))
        positive = sum(row["nested_diagnostic"]["mean_gain_nats_per_trial"] > 0 for row in rows)
        result.update(participants=108, combined_heldout_trials_per_participant=256,
            score_normalization="equal_mean_of_four64_trial_run_means_within_participant",
            participant_scores=scores, pooled_nested_diagnostic=loo.fixed.nested_contrast(scores, "MZ", "M0"),
            positive_run_mean_count=positive, at_least_three_of_four_run_means_positive=positive >= 3)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-id", type=parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != loo.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.source_root, args.job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "complete_held_out_runs", "unknown_held_out_runs",
          "pooled_diagnostic_complete")}, indent=2), flush=True)
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
