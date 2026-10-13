#!/usr/bin/env python3
"""Pair the two explicit nonlinear M0/MZ components on the same saved MZ0.

The fixed primary scores qualify the source and remain separate. Unequal gain
and loss powers make nonlinear M0 a curvature-adjusted reference; this is not
fixed-versus-nonlinear arbitration or a global symmetric-valuation claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_nonlinear as nonlinear


MODELS = ("M0", "MZ")
PHASE = "MZ0_nonlinear_value_M0_MZ_nested_diagnostic"
NORMALIZATION = "sum_of_two128_trial_heldout_log_likelihoods_divided_by256_per_participant"


def parent_job(value):
    try:
        job = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Supply the explicit positive parent job ID") from error
    if type(value) is bool or job < 1 or str(job) != str(value):
        raise argparse.ArgumentTypeError("Supply the explicit positive parent job ID")
    return job


def report_paths(source_root, job_id):
    job = parent_job(job_id)
    root = Path(source_root).resolve()
    if root != (nonlinear.SOURCE.parent / f"20261005-MZ-nonlinear-{job}").resolve():
        raise ValueError("Use the explicit selected parent-job nonlinear directory")
    return {model: root / model / nonlinear.REPORT_NAME for model in MODELS}


def duration(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Retain finite nonnegative process and fit durations")
    return value


def validate_component(report, model, job_id, spec):
    """Return status/cost evidence, nonlinear scores, primary scores and identity."""
    technical = report.get("status") == "technical_nonlinear_failure"
    directions = [{"training_runs": list(train), "scoring_runs": list(held)} for train, held in nonlinear.shared.HALVES]
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != nonlinear.PHASE
            or report.get("generating_model") != "MZ" or type(report.get("replicate_index")) is not int
            or report["replicate_index"] != 0 or report.get("seed") != nonlinear.shared.SEED
            or report.get("fitted_model") != model or report.get("rival") != "nonlinear"
            or str(report.get("job_id")) != str(job_id)
            or report.get("source_dir") != str(nonlinear.SOURCE)
            or report.get("source_report") != str(nonlinear.SOURCE / nonlinear.SOURCE_REPORT)
            or report.get("score_normalization") != NORMALIZATION):
        raise ValueError("Each nonlinear component must retain its selected parent, model and exact MZ0 source")
    if (any(report.get(field) != "not_evaluated" for field in nonlinear.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("episode_terminal") is not False
            or report.get("selected_replicates") != [0]
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_nonlinear", "observed_choices_or_RTs_used", "wave_selected", "automatic_retry"))
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= nonlinear.MAX_SECONDS):
        raise ValueError("Preserve the bounded diagnostic-only nonlinear component scope")
    if (not technical or "implementation" in report) and report.get("implementation") != spec:
        raise ValueError("Use the unchanged original500 specification, priors and gamma support")
    elapsed = duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.tzinfo is None or finished.tzinfo is None or finished < began:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Each supplied component must be a terminal recorded attempt") from error
    row = {"fitted_model": model, "job_id": str(job_id), "elapsed_seconds": elapsed, "finished_utc": report["finished_utc"]}
    fits, scored = report.get("fits"), report.get("scored_primary_directions")
    if (not isinstance(fits, list) or len(fits) > 2 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored) + 1
            or scored != directions[:len(scored)]):
        raise ValueError("Retain the ordered returned/scored primary-direction prefix")
    if technical:
        if (report.get("nonlinear_component_complete") is not False or not isinstance(report.get("error"), str)
                or not report["error"] or "participant_scores" in report):
            raise ValueError("A technical nonlinear component cannot supply completed scores")
        row.update(status="technical_unknown", error=report["error"])
    elif (report.get("status") != "nonlinear_component_completed" or report.get("nonlinear_component_complete") is not True
            or len(fits) != 2 or scored != directions):
        raise ValueError("A completed nonlinear component requires both validated scored halves")
    else:
        row["status"] = "complete_component"
    validated, replayed = report.get("source_baseline_validated"), report.get("saved_primary_scores_replayed")
    if type(validated) is not bool or type(replayed) is not bool:
        raise ValueError("Source validation and primary replay must be explicit")
    identity = None
    if validated:
        source_job, source_finish = str(report.get("source_job_id", "")), report.get("source_finished_utc")
        if (not source_job.isdecimal() or int(source_job) < 1 or not isinstance(source_finish, str) or not source_finish
                or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]
                or report.get("heldout_trials_per_direction_per_participant") != 128
                or report.get("combined_scored_trials_per_participant") != 256):
            raise ValueError("Validated sources require full108-person primary128/256 support")
        identity = (source_job, source_finish)
    elif (not technical or replayed or fits or report.get("source_job_id") is not None
            or report.get("source_finished_utc") is not None):
        raise ValueError("An unvalidated source cannot supply fitted or replayed evidence")
    primary = None
    if replayed:
        discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_score_replay_absolute_tolerance") != nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE
                or type(discrepancy) not in (int, float) or not math.isfinite(discrepancy)
                or not 0 <= discrepancy <= nonlinear.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("Preserve original fixed-score replay at1e-12")
        primary = nonlinear.fixed.score_panel({"participant_scores": report.get("saved_primary_participant_scores")}, MODELS)
    elif not technical or fits or "saved_primary_participant_scores" in report:
        raise ValueError("New nonlinear fits require successful original primary replay")
    returned = []
    for position, (fit, (train, held)) in enumerate(zip(fits, nonlinear.shared.HALVES)):
        if (fit.get("model") != model or fit.get("rival") != "nonlinear"
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)
                or fit.get("participants") != sorted(primary)):
            raise ValueError("Each returned fit must retain its actual model, primary direction and participants")
        consumed = duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            nonlinear.validate_nonlinear_fit(fit, model, train, held, spec, primary)
        elif fit.get("converged") is not False or position < len(scored) or not technical:
            raise ValueError("Only converged validated nonlinear fits can supply a scored direction")
        returned.append({"training_runs": list(train), "scoring_runs": list(held),
                         "converged": fit["converged"], "elapsed_seconds": consumed})
    if technical:
        row.update(returned_fits=returned, scored_primary_directions=scored)
        return row, None, primary, identity
    scores = nonlinear.fixed.score_panel(report, ("nonlinear_score",))
    if set(scores) != set(primary) or any(scores[person]["version"] != primary[person]["version"] for person in scores):
        raise ValueError("Nonlinear scores must retain every qualified source participant and task version")
    return row, scores, primary, identity


def pool(source_root, job_id, spec):
    paths = report_paths(source_root, job_id)
    if spec["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Preserve original500 numerical fitting")
    rows, panels, identity, primary = [], {}, None, None
    for model in MODELS:
        try:
            report = json.loads(paths[model].read_text())
        except FileNotFoundError:
            rows.append({"fitted_model": model, "status": "missing_unknown", "report": str(paths[model])})
            continue
        row, scores, saved_primary, source_identity = validate_component(report, model, job_id, spec)
        row["report"] = str(paths[model])
        rows.append(row)
        if source_identity is not None:
            if identity is not None and source_identity != identity:
                raise ValueError("Both components must share the exact original source worker and completion")
            identity = source_identity
        if saved_primary is not None:
            if primary is not None and saved_primary != primary:
                raise ValueError("Both components must retain the same fixed-primary score last bits")
            primary = saved_primary
        if scores is not None:
            panels[model] = scores
    complete = len(panels) == 2
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE, "generating_model": "MZ",
        "replicate_index": 0, "seed": nonlinear.shared.SEED, "rival": "nonlinear",
        "source_root": str(Path(source_root).resolve()), "job_id": str(job_id),
        "source_dir": str(nonlinear.SOURCE), "source_report": str(nonlinear.SOURCE / nonlinear.SOURCE_REPORT),
        "source_job_id": identity[0] if identity else None, "source_finished_utc": identity[1] if identity else None,
        "implementation": spec, "status": "nonlinear_pair_diagnostic_completed" if complete else "incomplete_nonlinear_components",
        "pooled_diagnostic_complete": complete, "complete_models": sorted(panels),
        "unknown_models": [model for model in MODELS if model not in panels], "components": rows,
        "selected_replicates": [0], "wave_selected": False, "automatic_retry": False,
        "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in nonlinear.CLAIM_FIELDS},
        "fixed_vs_nonlinear_arbitration": "not_evaluated",
        "interpretation": "Nonlinear MZ-minus-M0 compares starting bias conditional on the same flexible gain/loss curvature. Unequal powers make nonlinear M0 a curvature-adjusted reference, not globally symmetric valuation. Fixed-primary scores remain separate. This one saved-cohort nested diagnostic supplies no nonnested winner, family, recovery rate, interval, curtailment or gate verdict."}
    if primary is not None:
        result["fixed_primary_participant_scores"] = primary
    if complete:
        scores = {person: {"version": primary[person]["version"],
                           **{model: panels[model][person]["nonlinear_score"] for model in MODELS}}
                  for person in sorted(primary)}
        nonlinear.fixed.score_panel({"participant_scores": scores}, MODELS)
        result.update(participants=108, combined_heldout_trials_per_participant=256,
            score_normalization=NORMALIZATION, nonlinear_participant_scores=scores,
            nonlinear_nested_diagnostic=nonlinear.fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-id", type=parent_job, required=True)
    parser.add_argument("--implementation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require_compute()
    if args.implementation.resolve() != nonlinear.IMPLEMENTATION.resolve():
        raise ValueError("Use the unchanged original implementation path")
    if args.output.exists():
        raise FileExistsError("Use a fresh summary output")
    result = pool(args.source_root, args.job_id, json.loads(args.implementation.read_text()))
    result["recorded_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "complete_models", "unknown_models", "pooled_diagnostic_complete")}, indent=2), flush=True)
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
