"""EP06 selected-electrode-unit and elapsed-time controls, development only.

One exclusive fit stage publishes all available unit-restricted predictions
and all fourteen time-baseline predictions before the evaluation reader exists.
Explicit unit-count/rank insufficiency leaves that entire control unavailable,
without dropping a session or preventing the separate timing comparison.
Original fit/score artifacts are trusted local pickles under fixed paths.
No calibration, reserved sessions, sibling outputs or source files are read.
"""
import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np
import bandwise_prediction as prediction
from component_scoring import BoundActivity
from development_views import CandidateDevelopmentViews, DevelopmentOutcomeViews
from electrode_unit_intersection import (IntersectionUnavailable, plan_paired_unit_intersection,
    restrict_paired_features, restrict_paired_activity)
from elapsed_time_baseline import (FittedElapsedBaseline, fit_elapsed_time_baseline,
    predict_elapsed_time_baseline, score_elapsed_time_baseline)
from primary_profile import paired_profile, score_band_profile
import run_initial_profiles as initial


ATTEMPT = initial.ATTEMPT_PATH.parent / "support001"
FAMILY = "starting_09"
RECIPE = dict(initial.FAMILIES)[FAMILY]
BASELINE = initial.CANDIDATES[10]


def _attempt(path):
    path = Path(path)
    if not path.is_absolute() or path != ATTEMPT or path.resolve() != ATTEMPT:
        raise PermissionError("only the declared support001 attempt is permitted")
    return path


def _program():
    return initial._plain({"kind": "EP06_selected_unit_and_elapsed_time_controls",
        "sessions": [asdict(spec) for spec in initial.SESSIONS],
        "recipe": asdict(RECIPE), "baseline": asdict(BASELINE),
        "controls": ["control04_selected_electrode_units", "control05_elapsed_time"],
        "unit_rule": "all_native_units_on_existing_selected15_LFP_electrodes",
        "elapsed_penalty": .01, "calibration_values_used": False,
        "reserved_sessions_opened": False})


def _original(views_path, check):
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as handle:
        original = json.load(handle)
    if original != {"program": initial._program(), "views_path": str(views_path)}:
        raise ValueError("support controls must retain the exact original development view/program")
    with initial._owned_read(initial.ATTEMPT_PATH / "fit_complete.json") as handle:
        if json.load(handle).get("completed_cells") != 84:
            raise PermissionError("original fits must be complete")
    cells = {}
    with initial._owned_read(initial.ATTEMPT_PATH / f"{FAMILY}.pickle", binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                cell = pickle.load(stream)
                initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                cells[spec.filename, region] = cell
        if stream.read(1):
            raise ValueError("unexpected original CCA cells")
    return cells


def _validate_elapsed(cell, spec, region, check):
    check()
    if set(cell) != {"session", "region", "fit", "prediction"} or (
            cell["session"], cell["region"]) != (spec.filename, region):
        raise ValueError("exact ordered time-baseline cell required")
    fit, predicted = cell["fit"], cell["prediction"]
    if not isinstance(fit, FittedElapsedBaseline) or not isinstance(predicted, BoundActivity):
        raise TypeError("one genuine elapsed fit and bound prediction required")
    context = fit.source_context
    if (context.recipe != RECIPE or context.region != region
            or context.training_design.session_id != spec.filename
            or predicted.design.session_id != spec.filename or predicted.design.role != "evaluation"
            or predicted.design.axis_ids != context.population_axis_ids):
        raise ValueError("elapsed cell changed its original population context or outcome role")
    native_design = replace(predicted.design, axis_ids=context.training_design.axis_ids)
    if prediction._bind_fitted(context, native_design, region) != predicted.design:
        raise ValueError("elapsed prediction does not bind the frozen population map")
    values = np.asarray(predicted.values)
    if values.shape != predicted.design.shape or not np.isfinite(values).all():
        raise ValueError("complete finite elapsed predictions required")


def fit_predict(*, views_path, attempt_dir=ATTEMPT, check):
    initial._compute_only()
    check()
    attempt = _attempt(attempt_dir)
    views_path = Path(views_path)
    original = _original(views_path, check)
    attempt.mkdir(mode=0o700)
    initial._json_new(attempt / "recipes.json", {"program": _program(), "views_path": str(views_path)})
    views = CandidateDevelopmentViews(views_path, check=check)
    plans, metadata, support_errors = {}, {}, []
    for spec in initial.SESSIONS:
        for role in ("training", "evaluation"):
            for region in initial.REGIONS:
                metadata[spec.filename, region, role] = views.metadata(spec.filename, region, role)
        try:
            plans[spec.filename] = plan_paired_unit_intersection(
                metadata[spec.filename, "M1", "training"], metadata[spec.filename, "PMd", "training"], check=check)
        except IntersectionUnavailable as error:
            support_errors.append({"session": spec.filename, "reason": str(error)})
    initial._json_new(attempt / "unit_support.json", {"plans": [asdict(plan) for plan in plans.values()],
        "unavailable": support_errors, "rule": "units_on_selected_LFP_electrodes_not_bidirectional_intersection"})
    with initial._new_file(attempt / "unit_plans.pickle", binary=True) as stream:
        initial._dump_cell(stream, plans)
    restricted_available = not support_errors
    restriction_reason = None if restricted_available else "at_least_one_region_has_fewer_than_two_selected_units"
    restricted_count = elapsed_count = 0
    with initial._new_file(attempt / "fit_status.jsonl") as status:
        initial._event(status, "fit_started", job_id=os.environ.get("SLURM_JOB_ID"))
        try:
            with initial._new_file(attempt / "selected_units.pickle", binary=True) as unit_stream, \
                 initial._new_file(attempt / "elapsed_time.pickle", binary=True) as time_stream:
                for spec in initial.SESSIONS:
                    train = [views.training(spec.filename, region) for region in initial.REGIONS]
                    if restricted_available:
                        plan = plans[spec.filename]
                        lfp = restrict_paired_features(train[0][0], train[1][0], plan=plan, check=check)
                        counts = restrict_paired_activity(train[0][1], train[1][1], plan=plan, check=check)
                        evaluation = [views.features(spec.filename, region, "evaluation") for region in initial.REGIONS]
                        evaluation = restrict_paired_features(*evaluation, plan=plan, check=check)
                        for index, region in enumerate(initial.REGIONS):
                            try:
                                fitted = prediction.fit_bandwise(lfp[index], counts[index], recipe=RECIPE, check=check)
                            except ValueError as error:
                                if "population rank 2 exceeds numerical training rank" not in str(error):
                                    raise
                                restricted_available = False
                                restriction_reason = f"{spec.filename}/{region}: {error}"
                                initial._event(status, "unit_control_unavailable", reason=restriction_reason)
                                break
                            predicted = prediction.predict_bandwise(fitted, evaluation[index], check=check)
                            cell = {"family": FAMILY, "session": spec.filename, "region": region,
                                    "fit": fitted, "predictions": predicted}
                            initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                            initial._dump_cell(unit_stream, cell)
                            restricted_count += 1
                    for index, region in enumerate(initial.REGIONS):
                        fitted = fit_elapsed_time_baseline(original[spec.filename, region]["fit"],
                            metadata[spec.filename, region, "training"], train[index][1], check=check)
                        predicted = predict_elapsed_time_baseline(fitted,
                            metadata[spec.filename, region, "evaluation"], check=check)
                        cell = {"session": spec.filename, "region": region, "fit": fitted, "prediction": predicted}
                        _validate_elapsed(cell, spec, region, check)
                        initial._dump_cell(time_stream, cell)
                        elapsed_count += 1
                    initial._event(status, "session_predictions_published", session=spec.filename,
                        unit_cells=restricted_count, elapsed_cells=elapsed_count)
            if elapsed_count != 14 or (restricted_available and restricted_count != 14):
                raise RuntimeError("all available control predictions must be published")
            initial._json_new(attempt / "fit_complete.json", {"completed_utc": initial._utc(),
                "elapsed_cells": elapsed_count, "unit_cells": restricted_count,
                "unit_control_available": restricted_available, "unit_unavailable_reason": restriction_reason})
            initial._event(status, "fit_complete")
        except BaseException as error:
            initial._event(status, "fit_failed", error_type=type(error).__name__, error=str(error),
                           unit_cells=restricted_count, elapsed_cells=elapsed_count)
            raise


def _load_completed(attempt, check):
    check()
    with initial._owned_read(attempt / "fit_complete.json") as handle:
        complete = json.load(handle)
    if (complete["elapsed_cells"] != 14 or not isinstance(complete["unit_control_available"], bool)
            or (complete["unit_control_available"] and complete["unit_cells"] != 14)
            or (not complete["unit_control_available"] and not complete["unit_unavailable_reason"])):
        raise PermissionError("complete prediction publication or explicit whole-control unavailability required")
    with initial._owned_read(attempt / "recipes.json") as handle:
        declaration = json.load(handle)
    if declaration["program"] != _program():
        raise ValueError("support-control recipes changed")
    with initial._owned_read(initial.ATTEMPT_PATH / "recipes.json") as handle:
        if declaration["views_path"] != json.load(handle)["views_path"]:
            raise ValueError("original and support views differ")
    with initial._owned_read(attempt / "unit_plans.pickle", binary=True) as stream:
        plans = pickle.load(stream)
        if stream.read(1):
            raise ValueError("unexpected unit plans")
    elapsed, restricted = {}, {}
    with initial._owned_read(attempt / "elapsed_time.pickle", binary=True) as stream:
        for spec in initial.SESSIONS:
            for region in initial.REGIONS:
                cell = pickle.load(stream)
                _validate_elapsed(cell, spec, region, check)
                elapsed[spec.filename, region] = cell
        if stream.read(1):
            raise ValueError("unexpected timing cells")
    if complete["unit_control_available"]:
        if set(plans) != {spec.filename for spec in initial.SESSIONS}:
            raise ValueError("all seven paired unit plans required")
        with initial._owned_read(attempt / "selected_units.pickle", binary=True) as stream:
            for spec in initial.SESSIONS:
                for region in initial.REGIONS:
                    cell = pickle.load(stream)
                    initial._validate_cell(cell, FAMILY, RECIPE, spec, region, check=check)
                    regional_plan = plans[spec.filename].m1 if region == "M1" else plans[spec.filename].pmd
                    if cell["fit"].training_design.axis_ids != regional_plan.retained_axis_ids:
                        raise ValueError("restricted fit axes differ from the exact planned unit subset")
                    restricted[spec.filename, region] = cell
            if stream.read(1):
                raise ValueError("unexpected restricted cells")
    return Path(declaration["views_path"]), complete, plans, restricted, elapsed


def score(*, attempt_dir=ATTEMPT, check):
    initial._compute_only()
    attempt = _attempt(attempt_dir)
    views_path, completion, plans, restricted, elapsed = _load_completed(attempt, check)
    with initial._owned_read(initial.ATTEMPT_PATH / "scores/starting_10.json") as handle:
        baseline = json.load(handle)
    if baseline["candidate"] != initial._plain(asdict(BASELINE)):
        raise ValueError("exact original low-six comparator required")
    original_profiles = {row["session"]: row for row in baseline["profiles"]}
    if set(original_profiles) != {spec.filename for spec in initial.SESSIONS}:
        raise ValueError("all seven original scored panels required")
    destination = attempt / "scores"
    destination.mkdir(mode=0o700)
    with initial._new_file(destination / "status.jsonl") as status:
        initial._event(status, "score_started", job_id=os.environ.get("SLURM_JOB_ID"))
        try:
            outcomes = DevelopmentOutcomeViews(views_path, role="evaluation", check=check)
            paired, timing = {}, []
            for spec in initial.SESSIONS:
                observed = [outcomes.targets(spec.filename, region) for region in initial.REGIONS]
                if completion["unit_control_available"]:
                    narrowed = restrict_paired_activity(*observed, plan=plans[spec.filename], check=check)
                    regional = []
                    for index, region in enumerate(initial.REGIONS):
                        cell = restricted[spec.filename, region]
                        regional.append(score_band_profile(cell["fit"], narrowed[index], cell["predictions"], check=check))
                    paired[FAMILY, spec.filename] = paired_profile(*regional, check=check)
                for index, region in enumerate(initial.REGIONS):
                    cell = elapsed[spec.filename, region]
                    result = score_elapsed_time_baseline(cell["fit"], observed[index], cell["prediction"], check=check)
                    label = region.lower()
                    original_energy = original_profiles[spec.filename][f"{label}_residual_energy"]
                    if result.residual_energy is not None and not math.isclose(
                            result.residual_energy, original_energy, rel_tol=1e-12, abs_tol=0):
                        raise ValueError("timing and original LFP comparisons must use identical residual energy")
                    lfp_scores = original_profiles[spec.filename][f"{label}_scores"]
                    timing.append({"session": spec.filename, "animal": spec.animal, "region": region,
                        "score": result.score, "residual_energy": result.residual_energy,
                        "squared_error": result.squared_error, "direction_time_score": result.direction_time_score,
                        "unavailable_reason": result.unavailable_reason,
                        "original_lfp_scores": lfp_scores,
                        "lfp_minus_elapsed": [None if result.score is None or value is None else value-result.score for value in lfp_scores]})
            if completion["unit_control_available"]:
                units = initial.evaluate_candidate(paired, BASELINE, check=check)
                units["control_id"] = "control04_selected_electrode_units"
                units["baseline_directional_balanced_accuracy"] = baseline["directional_balanced_accuracy"]
                units["claim_scope"] = "changed_population_units_on_selected_LFP_electrodes_sensitivity_only"
                units_available = units["joint_label_reference"]["complete"] and units["signed_agreement"]["raw_available"]
            else:
                units = {"control_id": "control04_selected_electrode_units", "status": "unavailable",
                    "reason": completion["unit_unavailable_reason"], "scientific_acceptance": None}
                units_available = False
            initial._json_new(destination / "control04_selected_electrode_units.json", units)
            timing_available = len(timing) == 14 and all(row["score"] is not None for row in timing)
            initial._json_new(destination / "control05_elapsed_time.json", {
                "control_id": "control05_elapsed_time", "complete": timing_available,
                "blocks": [asdict(block) for block in initial.BLOCKS[:6]], "cells": timing,
                "claim_scope": "prediction_vs_fixed_linear_timing_baseline_not_conditional_information",
                "scientific_acceptance": None})
            initial._json_new(destination / "complete.json", {"completed_utc": initial._utc(),
                "available_controls": {"control04_selected_electrode_units": units_available,
                                       "control05_elapsed_time": timing_available},
                "valid_scientific_control_count": int(units_available) + int(timing_available),
                "reserved_sessions_opened": False, "calibration_values_used": False,
                "required_controls_passed": None, "scientific_acceptance": None})
            initial._event(status, "score_complete")
        except BaseException as error:
            initial._event(status, "score_failed", error_type=type(error).__name__, error=str(error))
            raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fit_predict", "score"), required=True)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    initial._compute_only()
    if not math.isfinite(args.deadline_epoch):
        parser.error("finite absolute deadline required")
    def check():
        if time.time() >= args.deadline_epoch:
            raise TimeoutError("coordinator-owned deadline reached")
    if args.mode == "fit_predict":
        if args.views is None:
            parser.error("fit_predict requires --views")
        fit_predict(views_path=args.views, check=check)
    else:
        if args.views is not None:
            parser.error("score uses its published view declaration")
        score(check=check)


if __name__ == "__main__":
    main()
