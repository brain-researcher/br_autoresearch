"""Complete-source EP19 scoring of existing development-fold predictions.

The caller independently supplies every scored natural-grid row and its labels,
after authentic risk/onset construction and self-paced prefix/guard exclusion.
All 72 WAY or five self-paced folds must bind before any score is returned.
This module performs no source I/O, fitting, condition construction, calibration,
recipe selection, margin choice or held-out/series7 evaluation.

Input folds remain caller-owned and must remain fixed throughout the call. The
complete prediction panel is additionally resident in memory; production memory
and runtime are unmeasured. Budget callbacks surround assembly and each existing
scorer invocation; the caller also owns the outer allocation/process limit.
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from classical_forecast_backend import FittingScope, _labels
from development_baseline_driver import baseline_recipes
from ensemble_forecast_backend import _row_keys
from forecast_scoring import (
    HORIZONS, ScoringScope, UnresolvedScore, _log_probabilities, score_matched_forecasts,
)
from model_recipes import reference_recipes, self_paced_folds, way_inner_folds


ADAPTATIONS = ("primary", "zero_label")


def _readonly(value):
    value.setflags(write=False)
    return value


def _fold_roster(registry, scope):
    roster = {}
    if scope.source == "WAY":
        for person in scope.participants:
            for fold in way_inner_folds(registry, person):
                fitting = FittingScope.development(registry, source="WAY",
                    participant=person, fold_id=fold.fold_id)
                roster[fold.fold_id] = (fitting, (person,), fold.validation)
    else:
        for fold in self_paced_folds(registry):
            fitting = FittingScope.development(registry, source="self_paced", fold_id=fold.fold_id)
            roster[fold.fold_id] = (fitting, fold.validation, None)
    if len(roster) != (72 if scope.source == "WAY" else 5):
        raise ValueError("complete_registered_development_fold_roster_required")
    return roster


def _expected_rows(registry, source, row_keys, observed_category, complete_bins,
                   observed_through_bins, check):
    scope = ScoringScope.from_registry(registry, source=source, stage="development")
    roster = _fold_roster(registry, scope)
    keys = _row_keys(row_keys)
    category, complete = _labels(observed_category, complete_bins, len(keys))
    _, coverage = _labels(observed_category, observed_through_bins, len(keys))
    if np.any((category >= 0) & (coverage < category)):
        raise ValueError("expected_onset_category_contradicts_declared_followup")
    destinations = {}
    positions = {}
    for fold_id, (_, people, series) in roster.items():
        for person in people:
            positions[fold_id, person] = []
            destinations[person, None if series is None else series[0]] = fold_id
    boundaries, last_times = {}, {}
    for index, (identity, timestamp) in enumerate(keys):
        check()
        destination = (identity.participant, identity.series_or_run if source == "WAY" else None)
        if identity.source != source or destination not in destinations:
            raise ValueError("expected_rows_cross_development_participant_or_series_role")
        group = (source, identity.participant, identity.series_or_run)
        if boundaries.setdefault(identity.boundary_id, group) != group:
            raise ValueError("one_reset_boundary_cannot_name_multiple_recording_groups")
        if timestamp <= last_times.get(identity.boundary_id, -np.inf):
            raise ValueError("expected_decisions_must_increase_within_each_reset_boundary")
        last_times[identity.boundary_id] = timestamp
        positions[destinations[destination], identity.participant].append(index)
    if any(not indices for indices in positions.values()):
        raise UnresolvedScore("every_registered_participant_validation_fold_requires_expected_rows")
    return scope, roster, keys, category, complete, coverage, positions


def _bind_folds(fold_results, roster, scope, keys, positions, baseline_id, reference_id, check):
    """Bind the entire roster and identities before inspecting prediction arrays."""
    folds = tuple(fold_results)
    if len(folds) != len(roster):
        raise UnresolvedScore("all_required_development_folds_required_no_partial_panel")
    bound = {}
    for result in folds:
        check()
        if not isinstance(result, Mapping):
            raise ValueError("existing_development_fold_prediction_result_required")
        fitting = result.get("fitting_scope")
        if not isinstance(fitting, FittingScope) or fitting.fold_id not in roster:
            raise ValueError("canonical_development_fitting_scope_required")
        fold_id = fitting.fold_id
        expected, people, _ = roster[fold_id]
        if fold_id in bound:
            raise UnresolvedScore("duplicate_development_fold_is_not_replacement_for_missing_fold")
        if (fitting != expected or result.get("source") != scope.source
                or result.get("stage") != "development_fold_validation_predictions"
                or result.get("completed_validation_fold") is not True
                or result.get("complete_source_panel") is not False
                or result.get("aggregate_feedback_produced") is not False
                or tuple(result.get("participants", ())) != people
                or tuple(result.get("conditions", ())) != scope.conditions
                or result.get("baseline_recipe_id") != baseline_id
                or result.get("reference_recipe_id") != reference_id):
            raise ValueError("fold_role_recipe_condition_or_completion_binding_changed")
        predictions = result.get("participant_predictions")
        if not isinstance(predictions, Mapping) or set(predictions) != set(people):
            raise UnresolvedScore("all_exact_validation_people_required_in_each_fold")
        for person in people:
            item = predictions[person]
            if not isinstance(item, Mapping):
                raise ValueError("existing_participant_prediction_result_required")
            row_keys = _row_keys(item.get("row_keys", ()))
            expected_keys = tuple(keys[index] for index in positions[fold_id, person])
            if row_keys != expected_keys:
                raise ValueError("fold_rows_must_equal_independently_supplied_scored_row_order")
            if (not isinstance(item.get("predictions"), Mapping)
                    or set(item["predictions"]) != set(scope.conditions)
                    or not isinstance(item.get("calibration_reports"), Mapping)
                    or set(item["calibration_reports"]) != set(scope.conditions)):
                raise UnresolvedScore("all_five_matched_predictions_and_calibration_records_required")
            for values in item["predictions"].values():
                if not isinstance(values, Mapping) or set(values) != set(ADAPTATIONS):
                    raise UnresolvedScore("both_existing_adaptation_outputs_required_for_every_condition")
        bound[fold_id] = result
    if set(bound) != set(roster):
        raise UnresolvedScore("every_canonical_development_fold_required")
    return bound


def _calibration_report(report, source):
    if source == "WAY":
        if report is not None:
            raise ValueError("WAY_development_predictions_must_be_unadapted")
        return None
    if (not isinstance(report, Mapping) or type(report.get("accepted")) is not bool
            or type(report.get("identity_fallback")) is not bool
            or report["accepted"] == report["identity_fallback"]):
        raise ValueError("original_self_paced_calibration_success_or_fallback_evidence_required")
    parameters = np.asarray(report.get("parameters"))
    if (np.iscomplexobj(parameters) or parameters.shape != (4,)
            or not np.issubdtype(parameters.dtype, np.number)
            or not np.isfinite(parameters).all() or parameters[0] <= 0
            or (report["identity_fallback"] and not np.array_equal(parameters, [1., 0., 0., 0.]))):
        raise ValueError("unchanged_four_parameter_calibration_or_identity_fallback_required")
    return deepcopy(dict(report))


def score_reference_development_panel(fold_results, *, registry, source,
        baseline_recipe_id, reference_recipe_id, expected_row_keys, observed_category,
        complete_bins, observed_through_bins, budget_check):
    """Assemble one recipe's COMPLETE source, then reuse all three horizon scores.

    Expected rows/labels are supplied independently of fold outputs, including
    every eligible scored suffix row, not inferred from successfully returned
    folds. Input order between folds may vary; each person's row order within a
    fold must equal its projection from the expected roster. Fitting scopes and
    recipe identities must match; fold-specific fitted parameters may differ.

    Missing primary-horizon support refuses the source. Other horizon support
    can remain explicitly unresolved, without dropping a person. Self-paced
    primary is calibrated32 and zero_label is its paired unadapted suffix; both
    WAY keys are the producer's identical unadapted predictions, not two fits.
    Original calibration reports, including identity fallbacks, are retained.
    This reports continuous inner-fold evidence only: it neither selects a
    recipe nor consumes/releases series7 feedback or any held-out role.
    """
    if not callable(budget_check):
        raise ValueError("explicit_allocation_owned_budget_callback_required")
    budget_check()
    scope, roster, keys, category, complete, coverage, positions = _expected_rows(
        registry, source, expected_row_keys, observed_category, complete_bins,
        observed_through_bins, budget_check)
    if baseline_recipe_id not in {recipe.recipe_id for recipe in baseline_recipes(registry, source=source)}:
        raise ValueError("registered_source_baseline_recipe_required")
    if reference_recipe_id not in {recipe.recipe_id for recipe in reference_recipes(registry)}:
        raise ValueError("registered_reference_recipe_required")
    bound = _bind_folds(fold_results, roster, scope, keys, positions,
                        baseline_recipe_id, reference_recipe_id, budget_check)
    predictions = {adaptation: {name: np.empty((len(keys), 19), dtype=np.float64)
                               for name in scope.conditions} for adaptation in ADAPTATIONS}
    calibrations, fallbacks = [], []
    for fold_id, (_, people, _) in roster.items():
        budget_check()
        for person in people:
            budget_check()
            item = bound[fold_id]["participant_predictions"][person]
            indices = np.asarray(positions[fold_id, person], dtype=np.intp)
            local_category, local_complete = _labels(item.get("observed_category"),
                                                     item.get("complete_bins"), len(indices))
            _, local_coverage = _labels(item.get("observed_category"),
                                        item.get("observed_through_bins"), len(indices))
            if (not np.array_equal(local_category, category[indices])
                    or not np.array_equal(local_complete, complete[indices])
                    or not np.array_equal(local_coverage, coverage[indices])):
                raise ValueError("fold_labels_must_equal_independently_supplied_labels_by_row_identity")
            reports = {}
            for name in scope.conditions:
                budget_check()
                values = item["predictions"][name]
                if source == "WAY" and not np.array_equal(values["primary"], values["zero_label"]):
                    raise ValueError("WAY_primary_and_zero_label_must_be_identical_unadapted_predictions")
                for adaptation in ADAPTATIONS:
                    predictions[adaptation][name][indices] = _log_probabilities(values[adaptation], len(indices))
                report = _calibration_report(item["calibration_reports"][name], source)
                reports[name] = report
                if report is not None and report["identity_fallback"]:
                    fallbacks.append({"fold_id": fold_id, "participant": person, "condition": name})
            calibrations.append({"fold_id": fold_id, "participant": person, "reports": reports})
    for values in predictions.values():
        for value in values.values():
            _readonly(value)
    scores = {}
    for adaptation in ADAPTATIONS:
        scores[adaptation] = {}
        for horizon in HORIZONS:
            budget_check()
            try:
                result = score_matched_forecasts(registry=registry, scope=scope, horizon=horizon,
                    expected_row_keys=keys, condition_log_probabilities=predictions[adaptation],
                    condition_row_keys={name: keys for name in scope.conditions},
                    observed_category=category, observed_through_bins=coverage)
            except UnresolvedScore:
                if horizon == "primary_strict":
                    raise
                scores[adaptation][horizon] = {"complete": False, "result": None,
                    "unresolved_reason": "not_every_registered_participant_has_observed_horizon_support"}
            else:
                scores[adaptation][horizon] = {"complete": True, "result": result,
                                              "unresolved_reason": None}
            budget_check()
    return {"source": source, "stage": "development_complete_inner_fold_scores",
        "participants": scope.participants, "fold_ids": tuple(roster), "fold_count": len(roster),
        "baseline_recipe_id": baseline_recipe_id, "reference_recipe_id": reference_recipe_id,
        "conditions": scope.conditions, "row_keys": keys, "observed_category": _readonly(category),
        "complete_bins": _readonly(complete), "observed_through_bins": _readonly(coverage),
        "predictions": predictions, "scores": scores, "calibration_reports": tuple(calibrations),
        "calibration_identity_fallbacks": tuple(fallbacks), "complete_source_panel": True,
        "adaptation_modes": {"primary": "unadapted" if source == "WAY" else "calibrated32",
                             "zero_label": "unadapted" if source == "WAY" else "zero_label"},
        "source_admission_established": False, "calibration_prefix_qualification_established": False,
        "recipe_selected": False, "scientific_decision_made": False,
        "feedback_series7_evaluated": False, "held_out_evaluated": False}
