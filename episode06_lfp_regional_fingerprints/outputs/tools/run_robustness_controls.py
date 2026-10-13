"""Four fixed EP06 robustness controls using completed physical score artifacts.

Two controls omit each whole paired development session, refit both remaining
source classifiers, and score the remaining opposite-animal sessions. These
are influence sensitivities, not predictions of the omitted session. Each
source omission recomputes magnitude median/MAD from that source alone. Every
omission slot remains present; an unavailable fit makes its entire control
unavailable without preventing the other controls from being reported.

Two further controls use the same fixed five bands and prototype classifier
on existing selected-unit and trial-mismatch physical scores. Their complete
128-assignment references and raw signed diagnostics compare descriptively
with adaptive03. No new neural fit/outcome read, classifier candidate, final
opening, patience decision or stopping authority is introduced.
"""
import argparse
from dataclasses import asdict
import math
import os
from pathlib import Path
import time

import profile_classifier as classifiers
import profile_references as references
import magnitude_classifier as magnitude
import run_initial_profiles as initial
import run_magnitude_controls as reuse


ATTEMPT = initial.ATTEMPT_PATH.parent / "robustness001"
LOW5 = (0, 2, 3, 4, 5)
CONTROLS = (
    ("control10lmpleaveout", "control07lmp", "magnitude", (0,)),
    ("control11low5leaveout", "adaptive03", "prototype", LOW5),
    ("control12low5selectedunits", "adaptive03", "selected_units", LOW5),
    ("control13low5mismatch", "adaptive03", "mismatch", LOW5),
)


def _program():
    return initial._plain({"kind": "EP06_four_fixed_robustness_controls",
        "controls": [{"control_id": name, "comparator": baseline,
                      "operation": operation, "block_indices": indices}
                     for name, baseline, operation, indices in CONTROLS],
        "sessions": initial.ROSTERS, "comparison_count": 4, "new_classifier_candidates": 0,
        "leaveout": "one_entire_paired_session_removed_then_both_source_classifiers_refitted",
        "magnitude_scaling": "recompute_source_median_and_1.4826_MAD_no_target_fitting_or_fallback",
        "omission_failure": "entire_control_unavailable_retain_all_seven_slots_no_available_fold_average",
        "leaveout_reference": "no_additional_128_assignment_reference",
        "new_neural_fits": 0, "new_outcome_reads": 0, "calibration_values_used": False,
        "reserved_sessions_opened": False, "local_stage_seconds_max": 600,
        "global_clock_reset": False, "scientific_acceptance": None, "search_stop_decision": None})


def _classify(panels, rosters, source, kind, check):
    animal, target = references.ANIMALS[source], references.ANIMALS[1-source]
    labels = references._labels(animal, rosters[source], (0,)*len(rosters[source]))
    if kind == "magnitude":
        fit_function, predict_function, recipe = (magnitude.fit_magnitude_classifier,
            magnitude.predict_magnitude_classifier, initial.LOGISTIC1)
    elif kind == "prototype":
        fit_function, predict_function, recipe = (classifiers.fit_profile_classifier,
            classifiers.predict_profile_classifier, initial.PROTOTYPE)
    else:
        raise ValueError("only the fixed singleton magnitude or five-band prototype omission is admitted")
    fitted = fit_function(panels[source], source_animal=animal, expected_sessions=rosters[source],
                         labels=labels, recipe=recipe, check=check)
    predicted = predict_function(fitted, panels[1-source], target_animal=target,
                                 expected_sessions=rosters[1-source], check=check)
    score = classifiers.score_profile_predictions(predicted,
        labels=references._labels(target, rosters[1-source], (0,)*len(rosters[1-source])), check=check)
    fit_report = asdict(fitted)
    for field in ("coefficients", "prototypes"):
        if field in fit_report and fit_report[field] is not None:
            fit_report[field] = getattr(fitted, field).tolist()
    return {"source_animal": animal, "target_animal": target,
        "source_sessions": list(rosters[source]), "target_sessions": list(rosters[1-source]),
        "balanced_accuracy": score.balanced_accuracy, "tied_rows": score.tied_rows,
        "class_0_recall": score.class_0_recall, "class_1_recall": score.class_1_recall,
        "fit": fit_report, "predictions": [asdict(row) for row in predicted.rows]}


def evaluate_leaveout(panels, *, kind, check):
    """All seven paired omissions, with independent source fitting per direction."""
    panels, rosters, blocks, missing = references._panel(*panels, *initial.ROSTERS, check)
    expected = tuple(initial.BLOCKS[index] for index in ((0,) if kind == "magnitude" else LOW5))
    if kind not in ("magnitude", "prototype") or (blocks and blocks != expected):
        raise ValueError("the omission control must retain its fixed physical blocks and representation")
    slots = [{"omitted_animal": animal, "omitted_session": session, "directions": [None, None],
        "complete": False, "equal_direction_mean_balanced_accuracy": None, "unavailable_reason": None}
        for animal, roster in zip(references.ANIMALS, rosters) for session in roster]
    result = {"complete": False, "fit_attempts": 0, "unavailable_reason": missing,
        "omissions": slots, "scientific_control_count": 1,
        "interpretation": "whole_session_influence_not_held_out_omitted_session_prediction"}
    if missing is not None:
        for slot in slots:
            slot["unavailable_reason"] = missing
        return result
    offset = 0
    for omitted_animal, panel in enumerate(panels):
        for omitted_index in range(len(panel)):
            check()
            slot = slots[offset+omitted_index]
            reduced_panels, reduced_rosters = list(panels), list(rosters)
            reduced_panels[omitted_animal] = panel[:omitted_index] + panel[omitted_index+1:]
            roster = rosters[omitted_animal]
            reduced_rosters[omitted_animal] = roster[:omitted_index] + roster[omitted_index+1:]
            for source in range(2):
                result["fit_attempts"] += 1
                try:
                    slot["directions"][source] = _classify(reduced_panels, reduced_rosters, source, kind, check)
                except classifiers.ClassificationUnavailable as error:
                    slot["unavailable_reason"] = result["unavailable_reason"] = str(error)
                    for pending in slots[offset+omitted_index+1:]:
                        pending["unavailable_reason"] = "not_attempted_after_unavailable_omission"
                    return result
            slot["complete"] = True
            slot["equal_direction_mean_balanced_accuracy"] = .5*sum(
                direction["balanced_accuracy"] for direction in slot["directions"])
        offset += len(panel)
    result["complete"] = True
    return result


def _baselines(original, check):
    manifest = reuse._read_json(reuse.ATTEMPT / "recipes.json")
    if manifest.get("program") != reuse._program():
        raise ValueError("fixed completed magnitude comparison declaration changed")
    complete = reuse._read_json(reuse.ATTEMPT / "scores/complete.json")
    if (not complete.get("completed_utc") or complete.get("controls_attempted") != 4
            or complete.get("successors_attempted") != 2
            or [row["candidate_id"] for row in complete.get("comparisons", ())]
                != [value.candidate_id for value in reuse.COMPARISONS]
            or complete.get("calibration_values_used") is not False
            or complete.get("reserved_sessions_opened") is not False):
        raise PermissionError("the complete prior six-comparison stage must precede robustness scoring")
    reports, panels = {}, {}
    for name in ("control07lmp", "adaptive03"):
        report = reuse._read_json(reuse.ATTEMPT / f"scores/{name}.json")
        comparison = next(value for value in reuse.COMPARISONS if value.candidate_id == name)
        recipe = initial.LOGISTIC1 if name == "control07lmp" else initial.PROTOTYPE
        if report["candidate"] != initial._plain({**asdict(comparison), "classifier": asdict(recipe)}):
            raise ValueError("robustness comparators must be the exact completed representations")
        panels[name] = reuse._panels(original, comparison.block_indices, check)
        summaries = [initial._profile_summary(row) for panel in panels[name] for row in panel]
        if report["profiles"] != initial._plain(summaries):
            raise ValueError("baseline report and reused original physical scores differ")
        signed = report["signed_agreement"]
        if (tuple(signed["mihili_sessions"]) != initial.ROSTERS[0]
                or tuple(signed["chewie_sessions"]) != initial.ROSTERS[1]
                or [(row["omitted_animal"], row["omitted_session"]) for row in signed["omissions"]]
                    != [(animal, session) for animal, roster in zip(references.ANIMALS, initial.ROSTERS)
                        for session in roster]):
            raise ValueError("baseline raw signed diagnostics must preserve all seven session omissions")
        reports[name] = report
    return reports, panels


def _nominal(report, baseline, baseline_id):
    before, after = baseline["directional_balanced_accuracy"], report["directional_balanced_accuracy"]
    return {"candidate_id": baseline_id, "directional_balanced_accuracy": before,
        "directional_accuracy_change": [None if a is None or b is None else a-b for a, b in zip(after, before)],
        "claim_scope": "descriptive_same_representation_sensitivity_not_control_pass"}


def run(*, attempt_dir=ATTEMPT, deadline_epoch):
    initial._compute_only()
    if not math.isfinite(deadline_epoch):
        raise ValueError("finite coordinator-owned absolute deadline required")
    effective_deadline = min(deadline_epoch, time.time()+600.)
    def check():
        if time.time() >= effective_deadline:
            raise TimeoutError("shared deadline or 600-second robustness stage cap reached")
    check()
    attempt = Path(attempt_dir)
    if not attempt.is_absolute() or attempt != ATTEMPT or attempt.resolve() != ATTEMPT:
        raise PermissionError("only the fixed robustness001 attempt is permitted")
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "recipes.json", {"program": _program(),
        "coordinator_deadline_epoch": deadline_epoch, "effective_deadline_epoch": effective_deadline})
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)
    with initial._new_file(destination / "status.jsonl") as status:
        initial._event(status, "robustness_started", job_id=os.environ.get("SLURM_JOB_ID"))
        summaries = []
        try:
            reuse._completed_sources(check)
            original = reuse._score_stream("original", [name for name, _ in initial.FAMILIES], "starting_09", check)
            baselines, panels = _baselines(original, check)
            for name, baseline_id, operation, indices in CONTROLS:
                check()
                baseline = baselines[baseline_id]
                if operation in ("magnitude", "prototype"):
                    leaveout = evaluate_leaveout(panels[baseline_id], kind=operation, check=check)
                    report = {"control_id": name, "baseline_id": baseline_id,
                        "block_indices": indices, "classifier": baseline["candidate"]["classifier"],
                        "baseline_directional_balanced_accuracy": baseline["directional_balanced_accuracy"],
                        "signed_agreement_original": baseline["signed_agreement"],
                        "signed_agreement_source": str(reuse.ATTEMPT / f"scores/{baseline_id}.json"),
                        "leave_one_out": leaveout, "complete": leaveout["complete"],
                        "joint_label_reference": None,
                        "claim_scope": "seven_whole_paired_session_omissions_one_influence_control_not_crossvalidation",
                        "scientific_acceptance": None, "search_stop_decision": None,
                        "patience_reset_authorized": False}
                    available = leaveout["complete"]
                else:
                    control_id = ("control04_selected_electrode_units" if operation == "selected_units"
                                  else "control03_mismatch")
                    saved = reuse._read_json(reuse.SOURCE_DIRS[operation] / f"scores/{control_id}.json")
                    if saved.get("control_id") != control_id:
                        raise ValueError("reused physical control identity changed")
                    regional = reuse._control_profiles(operation, saved, check)
                    comparison = reuse.Comparison(name, "control", operation, "starting_09", indices,
                        "centered_unit_l2_shape_prototype", "adaptive03")
                    report = reuse._evaluate(comparison, regional, baseline, check)
                    report["control_id"] = name
                    report["nominal_comparator"] = _nominal(report, baseline, "adaptive03")
                    available = report["joint_label_reference"]["complete"] and report["signed_agreement"]["raw_available"]
                initial._json_new(destination / f"{name}.json", report)
                summaries.append({"control_id": name, "baseline_id": baseline_id, "available": available})
                initial._event(status, "control_published", control_id=name, available=available)
            check()
            initial._json_new(destination / "complete.json", {"completed_utc": initial._utc(),
                "controls": summaries, "controls_attempted": 4,
                "valid_scientific_control_count": sum(row["available"] for row in summaries),
                "new_classifier_candidates": 0, "new_neural_fits": 0, "new_outcome_reads": 0,
                "calibration_values_used": False, "reserved_sessions_opened": False,
                "required_controls_passed": None, "scientific_acceptance": None,
                "search_stop_decision": None, "patience_reset_authorized": False})
            initial._event(status, "robustness_complete", completed_controls=4)
        except BaseException as error:
            initial._event(status, "robustness_failed", completed_controls=len(summaries),
                           error_type=type(error).__name__, error=str(error))
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, default=ATTEMPT)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    run(attempt_dir=args.attempt_dir, deadline_epoch=args.deadline_epoch)


if __name__ == "__main__":
    main()
