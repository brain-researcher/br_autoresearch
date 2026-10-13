#!/usr/bin/env python3
"""Pool both explicit common-gain directions using total log sums/counts.

All108 participants retain equal weight after each participant's held-out log
sums are divided by that participant's96 EI or176 ER retained trials. Neither
half means nor full-range256-trial denominators enter this sensitivity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from qualify_assets import require_compute
import run_mz_common_gain_range as common
import summarize_mz_fixed_menu as fixed
from summarize_mz_nonlinear import duration, parent_job


PHASE = "MZ0_common_gain_10_20_both_directions_nested_diagnostic"


def report_paths(source_root, job_id):
    job = parent_job(job_id)
    root = Path(source_root).resolve()
    if root != (common.SOURCE.parent / f"20261005-MZ-common-gain-range-{job}").resolve():
        raise ValueError("Use the explicit selected common-gain parent directory")
    return {direction: root / f"dir{direction}" / common.REPORT_NAME for direction in (0, 1)}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def slot_set(slots, runs, count):
    if (type(count) is not int or count < 1 or not isinstance(slots, list) or len(slots) != count
            or any(not isinstance(slot, list) or len(slot) != 2 or any(type(value) is not int for value in slot)
                or slot[0] not in runs or not 1 <= slot[1] <= 64 for slot in slots)):
        raise ValueError("Retain each original training/held run-trial slot and actual nonzero count")
    result = {tuple(slot) for slot in slots}
    if len(result) != count:
        raise ValueError("Duplicate retained trial slots are prohibited")
    return result


def validate_support(support, primary, train, held):
    if not isinstance(support, dict) or set(support) != set(primary):
        raise ValueError("Retain the same108 source participants in the filtered support")
    for person, row in support.items():
        expected = 96 if primary[person]["version"] == "EI" else 176
        if (row.get("version") != primary[person]["version"] or row.get("full_session_trials") != expected):
            raise ValueError("Common gain10-through20 requires96 EI and176 ER full-session trials")
        training = slot_set(row.get("training_slots"), train, row.get("training_trials"))
        scoring = slot_set(row.get("heldout_slots"), held, row.get("heldout_trials"))
        if training & scoring or len(training | scoring) != expected:
            raise ValueError("Training/held slots must be disjoint and exhaust each retained participant's trials")


def validate_component(report, direction, job_id, spec):
    train, held = common.selected_runs(direction)
    technical = report.get("status") == "technical_common_gain_failure"
    if (report.get("analysis_id") != "behavioral_mechanism_kill_test" or report.get("phase") != common.PHASE
            or report.get("generating_model") != "MZ" or type(report.get("replicate_index")) is not int
            or report["replicate_index"] != 0 or report.get("seed") != common.shared.SEED
            or type(report.get("direction")) is not int or report["direction"] != direction
            or str(report.get("job_id")) != str(job_id) or report.get("training_runs") != list(train)
            or report.get("scoring_runs") != list(held) or report.get("fitted_models") != list(common.MODELS)
            or report.get("rival") != "fixed" or report.get("gain_range_dollars") != [10, 20]
            or report.get("filter_boundary_inclusive") is not True
            or report.get("source_dir") != str(common.SOURCE)
            or report.get("source_report") != str(common.SOURCE / common.SOURCE_REPORT)
            or report.get("score_normalization") != common.NORMALIZATION):
        raise ValueError("Preserve each actual selected direction, source, range and sum/count representation")
    if (any(report.get(field) != "not_evaluated" for field in common.CLAIM_FIELDS)
            or report.get("full_recovery_pass") is not None or report.get("selected_replicates") != [0]
            or any(report.get(field) is not False for field in ("simulation_regenerated", "simulation_copied",
                "saved_primary_fits_reused_for_common_gain", "observed_choices_or_RTs_used", "wave_selected",
                "automatic_retry", "task_versions_are_independent_replications", "episode_terminal"))
            or type(report.get("internal_seconds")) is not int or not 1 <= report["internal_seconds"] <= common.MAX_SECONDS):
        raise ValueError("Preserve bounded, fresh-fit, synthetic-only diagnostic scope")
    if (not technical or "implementation" in report) and report.get("implementation") != spec:
        raise ValueError("Both common-gain models retain the original500 scientific specification")
    elapsed = duration(report.get("elapsed_seconds"))
    try:
        began, finished = (datetime.fromisoformat(report[key]) for key in ("started_utc", "finished_utc"))
        if began.utcoffset() is None or finished.utcoffset() is None or finished < began:
            raise ValueError("Invalid terminal timestamps")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Only terminal explicit reports may be summarized") from error
    fits, scored, completed = report.get("fits"), report.get("scored_fits"), report.get("completed_model_scores")
    order = [{"model": model, "training_runs": list(train), "scoring_runs": list(held)} for model in common.MODELS]
    if (not isinstance(fits, list) or len(fits) > 2 or any(not isinstance(fit, dict) for fit in fits)
            or not isinstance(scored, list) or not len(scored) <= len(fits) <= len(scored) + 1
            or scored != order[:len(scored)] or not isinstance(completed, dict)
            or set(completed) != set(common.MODELS[:len(scored)])):
        raise ValueError("Retain exactly the returned/scored model prefix and its complete score evidence")
    row = {"direction": direction, "job_id": str(job_id), "elapsed_seconds": elapsed,
           "finished_utc": report["finished_utc"]}
    if technical:
        if (report.get("common_gain_component_complete") is not False or "participant_scores" in report
                or not isinstance(report.get("error"), str) or not report["error"]):
            raise ValueError("Technical components cannot supply final participant scores")
        row.update(status="technical_unknown", error=report["error"])
    elif (report.get("status") != "common_gain_component_completed" or report.get("common_gain_component_complete") is not True
            or len(fits) != 2 or scored != order):
        raise ValueError("A complete direction requires both valid fitted/scored models")
    else:
        row["status"] = "complete_component"
    validated, replayed = report.get("source_baseline_validated"), report.get("saved_primary_scores_replayed")
    if type(validated) is not bool or type(replayed) is not bool:
        raise ValueError("Full-source validation and primary replay must remain explicit")
    identity, primary = None, None
    if validated:
        job, finish = str(report.get("source_job_id", "")), report.get("source_finished_utc")
        if (not job.isdecimal() or int(job) < 1 or not isinstance(finish, str) or not finish
                or [report.get(field) for field in ("participants", "runs", "trials")] != [108, 432, 27648]):
            raise ValueError("The source must remain the full108-person original cohort")
        identity = job, finish
    elif not technical or replayed or fits or report.get("source_job_id") is not None or report.get("source_finished_utc") is not None:
        raise ValueError("Unvalidated sources cannot supply fitted or replayed evidence")
    if replayed:
        discrepancy = report.get("saved_score_replay_max_absolute_discrepancy")
        if (not validated or report.get("saved_score_replay_absolute_tolerance") != common.shared.REPLAY_ABSOLUTE_TOLERANCE
                or not finite(discrepancy) or not 0 <= discrepancy <= common.shared.REPLAY_ABSOLUTE_TOLERANCE):
            raise ValueError("The unchanged full-range source scores must replay at1e-12")
        primary = fixed.score_panel({"participant_scores": report.get("saved_primary_participant_scores")}, common.MODELS)
    elif not technical or fits or "saved_primary_participant_scores" in report:
        raise ValueError("Filtering and fitting require the original primary replay")
    support = report.get("retained_support")
    if support is not None:
        if not replayed:
            raise ValueError("Filtered support must follow source validation and replay")
        validate_support(support, primary, train, held)
    elif not technical or fits or completed:
        raise ValueError("Fitted or scored components require explicit retained support")
    returned = []
    for position, fit in enumerate(fits):
        model = common.MODELS[position]
        if (fit.get("model") != model or fit.get("rival") != "fixed" or fit.get("participants") != sorted(primary)
                or fit.get("training_runs") != list(train) or fit.get("scoring_runs") != list(held)):
            raise ValueError("Retain the actual training-only model/participant/direction identity")
        consumed = duration(fit.get("elapsed_seconds"))
        if fit.get("converged") is True:
            common.validate_fit(fit, model, train, held, spec, primary)
        elif fit.get("converged") is not False or position < len(scored) or not technical:
            raise ValueError("Scored models require converged original500 fits")
        returned.append({"model": model, "converged": fit["converged"], "elapsed_seconds": consumed})
    for model, panel in completed.items():
        if not isinstance(panel, dict) or set(panel) != set(primary):
            raise ValueError("Each completed score prefix must cover all108 participants")
        for person, score in panel.items():
            if (set(score) != {"log_sum", "trials"} or not finite(score["log_sum"])
                    or type(score["trials"]) is not int or score["trials"] != support[person]["heldout_trials"]):
                raise ValueError("Keep finite unnormalized held-out sums and exact actual counts")
    if technical:
        row.update(returned_fits=returned, scored_fits=scored, completed_model_scores=completed)
        return row, None, primary, identity, support
    scores = report.get("participant_scores")
    if not isinstance(scores, dict) or set(scores) != set(primary):
        raise ValueError("A completed common-gain direction needs all108 participant score rows")
    for person, score in scores.items():
        if (set(score) != {"version", "heldout_trials", "M0_log_sum", "MZ_log_sum"}
                or score["version"] != primary[person]["version"]
                or type(score["heldout_trials"]) is not int or score["heldout_trials"] != support[person]["heldout_trials"]
                or any(not finite(score[model + "_log_sum"]) or score[model + "_log_sum"] != completed[model][person]["log_sum"]
                       for model in common.MODELS)):
            raise ValueError("Final score rows must exactly retain completed sums, counts and task versions")
    return row, scores, primary, identity, support


def pool(source_root, job_id, spec):
    paths = report_paths(source_root, job_id)
    if spec["optimizer"]["maximum_sweeps"] != 500:
        raise ValueError("Preserve original500 fitting")
    rows, panels, supports, primary, identity = [], {}, {}, None, None
    for direction in (0, 1):
        try:
            report = json.loads(paths[direction].read_text())
        except FileNotFoundError:
            rows.append({"direction": direction, "status": "missing_unknown", "report": str(paths[direction])})
            continue
        row, scores, saved, source_identity, support = validate_component(report, direction, job_id, spec)
        row["report"] = str(paths[direction])
        rows.append(row)
        if source_identity is not None:
            if identity is not None and identity != source_identity:
                raise ValueError("Both directions must use the same original source completion")
            identity = source_identity
        if saved is not None:
            if primary is not None and primary != saved:
                raise ValueError("Both directions must preserve identical fixed-primary score last bits")
            primary = saved
        if support is not None:
            supports[direction] = support
        if scores is not None:
            panels[direction] = scores
    if len(supports) == 2:
        for person in primary:
            a, b = supports[0][person], supports[1][person]
            if (a["version"] != b["version"] or a["full_session_trials"] != b["full_session_trials"]
                    or {tuple(slot) for slot in a["training_slots"]} != {tuple(slot) for slot in b["heldout_slots"]}
                    or {tuple(slot) for slot in a["heldout_slots"]} != {tuple(slot) for slot in b["training_slots"]}):
                raise ValueError("Both directions must use exact complementary retained trial identities")
    complete = len(panels) == 2
    result = {"analysis_id": "behavioral_mechanism_kill_test", "phase": PHASE,
        "generating_model": "MZ", "replicate_index": 0, "seed": common.shared.SEED,
        "source_root": str(Path(source_root).resolve()), "job_id": str(job_id),
        "source_dir": str(common.SOURCE), "source_report": str(common.SOURCE / common.SOURCE_REPORT),
        "source_job_id": identity[0] if identity else None, "source_finished_utc": identity[1] if identity else None,
        "implementation": spec, "gain_range_dollars": [10, 20], "filter_boundary_inclusive": True,
        "status": "common_gain_nested_diagnostic_completed" if complete else "incomplete_common_gain_directions",
        "pooled_diagnostic_complete": complete, "complete_directions": sorted(panels),
        "unknown_directions": [direction for direction in (0, 1) if direction not in panels], "components": rows,
        "selected_replicates": [0], "wave_selected": False, "automatic_retry": False,
        "task_versions_are_independent_replications": False, "full_recovery_pass": None, "episode_terminal": False,
        **{field: "not_evaluated" for field in common.CLAIM_FIELDS},
        "interpretation": "A single saved-cohort common-gain sensitivity, using original500 training-only M0/MZ fits. Per-person log sums from both directions are divided by the total retained96 EI or176 ER trials; each of108 participants then has equal weight. Fixed full-range scores remain separate. This does not establish recovery, family, generality, template/missingness/heterogeneity robustness, intervals, curtailment or a gate verdict."}
    if primary is not None:
        result["fixed_primary_participant_scores"] = primary
    if complete:
        scores, totals = {}, {}
        for person in sorted(primary):
            count = sum(panels[direction][person]["heldout_trials"] for direction in (0, 1))
            expected = 96 if primary[person]["version"] == "EI" else 176
            if count != expected:
                raise ValueError("The two held-out counts must exactly exhaust each participant's retained trials")
            totals[person] = count
            scores[person] = {"version": primary[person]["version"],
                **{model: sum(panels[direction][person][model + "_log_sum"] for direction in (0, 1)) / count
                   for model in common.MODELS}}
        fixed.score_panel({"participant_scores": scores}, common.MODELS)
        result.update(participants=108, retained_trials_per_participant=totals,
            score_normalization="sum_of_both_heldout_log_sums_divided_by_total_actual_retained_trials_per_participant",
            common_gain_participant_scores=scores, common_gain_nested_diagnostic=fixed.nested_contrast(scores, "MZ", "M0"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--job-id", type=parent_job, required=True)
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
    print(json.dumps({key: result[key] for key in ("status", "complete_directions", "unknown_directions", "pooled_diagnostic_complete")}, indent=2), flush=True)
    return 0 if result["pooled_diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
