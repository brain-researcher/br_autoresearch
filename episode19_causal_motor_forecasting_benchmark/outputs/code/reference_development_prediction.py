"""Matched EP19 development-fold inference on already admitted prepared arrays.

One complete validation fold, one registered EEG recipe, four original EEG
conditions and their frozen baseline. This composes existing prediction,
probability averaging and four-parameter calibration; it fits no encoder,
constructs no condition and produces no score, ranking or aggregate feedback.
The caller must keep supplied arrays/models fixed and authenticate source
timing, natural-grid rows, condition construction, the first32-event prefix
and its history guard. This module opens no source and grants no data role.

All baseline offsets stay uncalibrated before EEG fusion. Prefix and suffix
inference reset independently, even inside one recording boundary. Guard rows
are not predicted or calibrated. Each model's one suffix evaluation supplies
both adaptation conditions. Failures propagate without partial results/retry.
Inputs and outputs remain memory resident; the caller owns resource limits.
"""
from collections.abc import Mapping

import numpy as np

from baseline_final_refit import FittedBaselineEnsemble
from classical_forecast_backend import (
    PastWindow, UnresolvedFit, _labels, canonical_reference_scope,
)
from development_baseline_driver import (
    CalibrationPartition, PreparedBaselineRow, PreparedParticipant,
    _integer, _predict as _predict_baseline, baseline_recipes,
)
from ensemble_forecast_backend import (
    EnsembleForecastModel, SEEDS, _row_keys, fit_ensemble_calibration,
    mean_seed_log_probabilities,
)
from forecast_components import calibrated_log_probabilities
from model_recipes import self_paced_folds, way_inner_folds
from non_neural_forecast_backend import ContextFeatures
from temporal_hazard_backend import ContextSequence


EEG_CONDITIONS = ("real_EEG", "zero_EEG", "shift_1", "phase_1")
CONDITIONS = ("non_neural", *EEG_CONDITIONS)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _fit_bindings(registry, baseline_fit, reference_fits, check):
    check()
    if (not isinstance(baseline_fit, Mapping) or baseline_fit.get("completed") is not True
            or baseline_fit.get("stage") != "development_baseline_inside_EEG_fitting_fold"
            or baseline_fit.get("fits_completed") != 3
            or tuple(baseline_fit.get("seeds", ())) != SEEDS):
        raise UnresolvedFit("complete_frozen_development_baseline_fit_required")
    baseline = baseline_fit.get("ensemble")
    if not isinstance(baseline, FittedBaselineEnsemble):
        raise ValueError("existing_fitted_baseline_ensemble_required")
    scope = canonical_reference_scope(registry, baseline.fitting_scope)
    recipes = {recipe.recipe_id: recipe for recipe in baseline_recipes(registry, source=scope.source)}
    recipe = baseline.recipe
    if (recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id]
            or baseline_fit.get("selected_recipe_id") != recipe.recipe_id
            or baseline_fit.get("source") != scope.source
            or tuple(registry["seeds"]) != SEEDS or recipe.seeds != SEEDS
            or len(baseline.models) != 3 or len(baseline.fit_reports) != 3):
        raise ValueError("one_registered_source_baseline_with_three_fitted_seeds_required")
    all_models = set()
    for seed, model, report in zip(SEEDS, baseline.models, baseline.fit_reports):
        check()
        if (not isinstance(report, Mapping) or getattr(model, "fitted", False) is not True
                or not (report.get("completed") is True or report.get("converged") is True)
                or getattr(model, "seed", None) != seed or report.get("seed") != seed
                or getattr(model, "recipe", None) != recipe or report.get("recipe_id") != recipe.recipe_id
                or getattr(model, "fitting_scope", None) != scope or report.get("fold_id") != scope.fold_id
                or id(model) in all_models):
            raise UnresolvedFit("exact_distinct_complete_baseline_seed_models_and_reports_required")
        all_models.add(id(model))
    if (tuple(registry["matched_conditions"]["development_surrogates"]) != EEG_CONDITIONS[2:]
            or not isinstance(reference_fits, Mapping) or set(reference_fits) != set(EEG_CONDITIONS)):
        raise UnresolvedFit("all_four_registered_development_EEG_conditions_required")
    references, common = {}, None
    for condition in EEG_CONDITIONS:
        check()
        fitted = reference_fits[condition]
        if (not isinstance(fitted, Mapping) or fitted.get("completed_scope") is not True
                or fitted.get("stage") != "development_reference_fitting_fold"
                or fitted.get("condition") != condition or fitted.get("fitting_scope") != scope
                or fitted.get("source") != scope.source or fitted.get("fits_completed") != 3
                or tuple(fitted.get("seeds", ())) != SEEDS
                or fitted.get("baseline_recipe_id") != recipe.recipe_id):
            raise UnresolvedFit("complete_same_fold_condition_fit_and_frozen_baseline_binding_required")
        ensemble = fitted.get("ensemble")
        if (not isinstance(ensemble, EnsembleForecastModel)
                or ensemble.fitting_scope != scope or ensemble.fitting_stage != "development"
                or set(ensemble.models) != set(SEEDS)):
            raise ValueError("existing_development_EEG_probability_ensemble_required")
        reports = tuple(fitted.get("fit_reports", ()))
        if len(reports) != 3 or any(not isinstance(report, Mapping) for report in reports):
            raise UnresolvedFit("all_three_original_EEG_fit_reports_required")
        by_seed = {report.get("seed"): report for report in reports}
        if set(by_seed) != set(SEEDS):
            raise UnresolvedFit("exact_three_EEG_seed_report_identities_required")
        # Reuse existing actual-model/report validation, without fitting or
        # changing any member. No report alone substitutes for a fitted model.
        verified = EnsembleForecastModel([(ensemble.models[seed], by_seed[seed]) for seed in SEEDS],
                                         registry=registry, fitting_scope=scope)
        filters = None
        if verified.recipe.family == "past_only_filterbank_bandpower_elastic_net":
            # The ensemble already binds all three members' existing FIRs.
            # Matched conditions must also retain that same preprocessing.
            coefficients = verified.models[SEEDS[0]].features.coefficients
            filters = tuple(tuple(np.asarray(value, dtype=np.float64)) for value in coefficients)
        signature = (verified.recipe, verified.rate, verified.adapter, filters)
        if (verified.adapter != "original" or fitted.get("selected_recipe_id") != verified.recipe.recipe_id
                or (common is not None and signature != common)):
            raise ValueError("all_four_conditions_require_the_same_original_recipe_and_analysis_grid")
        for model in verified.models.values():
            if id(model) in all_models:
                raise ValueError("each_condition_and_seed_requires_a_separately_fitted_model")
            all_models.add(id(model))
        common = signature
        references[condition] = verified
    return scope, baseline, references


def _prepare_validation(registry, scope, participants, windows, check):
    if scope.source == "WAY":
        folds = way_inner_folds(registry, scope.allowed_participants[0])
        fold = next(item for item in folds if item.fold_id == scope.fold_id)
        expected_people, expected_series = scope.allowed_participants, set(fold.validation)
    else:
        fold = next(item for item in self_paced_folds(registry) if item.fold_id == scope.fold_id)
        expected_people, expected_series = fold.validation, None
    people = tuple(participants)
    if (any(not isinstance(person, PreparedParticipant) for person in people)
            or len(people) != len(expected_people)
            or {person.participant for person in people} != set(expected_people)):
        raise UnresolvedFit("complete_canonical_validation_fold_required_no_replacement_or_subset")
    if not isinstance(windows, Mapping) or set(windows) != set(EEG_CONDITIONS):
        raise ValueError("explicit_windows_for_all_four_matched_conditions_required")
    by_person = {person.participant: person for person in people}
    prepared, all_keys, boundaries = {}, [], {}
    for participant in expected_people:
        check()
        person = by_person[participant]
        rows = tuple(person.rows)
        if not rows or any(not isinstance(row, PreparedBaselineRow) for row in rows):
            raise ValueError("complete_prepared_validation_rows_required")
        keys = _row_keys((row.identity, row.decision_time) for row in rows)
        closed, previous, last = set(), None, -np.inf
        for row in rows:
            check()
            identity = row.identity
            if (identity.source != scope.source or identity.participant != participant
                    or (expected_series is not None and identity.series_or_run not in expected_series)):
                raise ValueError("row_crosses_the_canonical_validation_role")
            group = (identity.source, participant, identity.series_or_run)
            if boundaries.setdefault(identity.boundary_id, group) != group:
                raise ValueError("one_reset_boundary_cannot_name_multiple_recording_groups")
            if identity.boundary_id != previous:
                if identity.boundary_id in closed:
                    raise ValueError("validation_boundaries_must_be_contiguous_and_chronological")
                closed.add(identity.boundary_id)
                previous, last = identity.boundary_id, -np.inf
            if row.decision_time <= last:
                raise ValueError("decision_times_must_increase_within_each_reset_boundary")
            last = row.decision_time
            if (not isinstance(row.context, ContextFeatures) or not isinstance(row.sequence, ContextSequence)
                    or row.context.source != scope.source or row.context.participant != participant
                    or row.context.boundary_id != identity.boundary_id
                    or row.sequence.source != scope.source or row.sequence.participant != participant):
                raise ValueError("context_inputs_must_bind_the_same_validation_identity")
        category, complete = _labels([row.observed_category for row in rows],
                                    [row.complete_bins for row in rows], len(rows))
        _, coverage = _labels(category, [row.observed_through_bins for row in rows], len(rows))
        if np.any((category >= 0) & (coverage < category)):
            raise ValueError("onset_category_contradicts_declared_followup")
        if scope.source == "WAY":
            if person.calibration is not None:
                raise ValueError("WAY_validation_is_unadapted")
            prefix_count, suffix_start, prefix_labels = 0, 0, None
        else:
            partition = person.calibration
            if (not isinstance(partition, CalibrationPartition)
                    or not _integer(partition.prefix_row_count) or not _integer(partition.scored_suffix_start)
                    or not 0 < partition.prefix_row_count <= partition.scored_suffix_start < len(rows)):
                raise ValueError("explicit_qualified_nonempty_prefix_guard_and_suffix_required")
            prefix_count, suffix_start = int(partition.prefix_row_count), int(partition.scored_suffix_start)
            prefix_labels = _labels(partition.prefix_observed_category,
                                    partition.prefix_complete_bins, prefix_count)
            maximum_risk = np.where(category[:prefix_count] >= 0, category[:prefix_count], complete[:prefix_count])
            if (np.any((prefix_labels[0] >= 0) & (prefix_labels[0] != category[:prefix_count]))
                    or np.any((prefix_labels[0] < 0) & (prefix_labels[1] > maximum_risk))):
                raise ValueError("prefix_labels_must_be_truncated_without_future_label_gain")
        active = (*range(prefix_count), *range(suffix_start, len(rows)))
        # Windows cover prediction rows only. The omitted guard has no window
        # read, prediction, likelihood contribution or calibration update.
        active_keys = tuple(keys[index] for index in active)
        all_keys.extend(active_keys)
        prepared[participant] = dict(rows=rows, keys=keys, category=category, complete=complete,
            coverage=coverage, prefix_count=prefix_count, suffix_start=suffix_start,
            prefix_labels=prefix_labels, active=active, active_keys=active_keys)
    expected_keys = set(_row_keys(all_keys))
    fixed_windows = {}
    for condition in EEG_CONDITIONS:
        supplied = windows[condition]
        if not isinstance(supplied, Mapping) or set(supplied) != expected_keys:
            raise ValueError("every_condition_window_must_join_exact_prefix_and_suffix_row_identities")
        fixed_windows[condition] = dict(supplied)
        if any(not isinstance(window, PastWindow) for window in supplied.values()):
            raise ValueError("existing_PastWindow_inputs_required_without_condition_construction")
    return tuple(expected_people), prepared, fixed_windows


def _mean(predictions, keys):
    return mean_seed_log_probabilities(predictions,
        seed_row_keys={seed: keys for seed in SEEDS}, expected_row_keys=keys)


def _predict_reference(model, rows, keys, windows, offsets, check):
    result = np.empty((len(rows), 19), dtype=np.float64)
    boundary = None
    for index, (row, key) in enumerate(zip(rows, keys)):
        check()
        if row.identity.boundary_id != boundary:
            boundary = row.identity.boundary_id
            model.reset(boundary_id=boundary)
        result[index] = model.predict_log_probabilities(windows[key], decision_time=row.decision_time,
                                                       context_log_probabilities=offsets[index])
    return result


def _adapt(prefix_predictions, suffix_predictions, item, check):
    start, count = item["suffix_start"], item["prefix_count"]
    suffix_keys = item["keys"][start:]
    zero_label = _mean(suffix_predictions, suffix_keys)
    if count:
        check()
        prefix_keys = item["keys"][:count]
        report = fit_ensemble_calibration(prefix_predictions,
            seed_row_keys={seed: prefix_keys for seed in SEEDS}, prefix_row_keys=prefix_keys,
            observed_category=item["prefix_labels"][0], complete_bins=item["prefix_labels"][1],
            weights=np.ones(count, dtype=np.float64))
        check()
        primary = calibrated_log_probabilities(zero_label, report["parameters"])
    else:
        primary, report = zero_label, None
    return {"primary": _readonly(primary), "zero_label": _readonly(zero_label)}, report


def predict_reference_development_fold(*, registry, baseline_fit, reference_fits,
        validation_participants, eeg_windows_by_condition, budget_check):
    """Return complete internal matched predictions for ONE development fold.

    ``baseline_fit`` and ``reference_fits`` are the existing development-fit
    results. The latter maps exactly real_EEG/zero_EEG/shift_1/phase_1 to one
    common recipe's separately fitted ensembles. ``validation_participants``
    uses existing PreparedParticipant/PreparedBaselineRow/CalibrationPartition.
    Each condition's window mapping is keyed by (GroupIdentity, decision_time)
    for exactly the used prefix and suffix rows; guard keys are excluded.

    The returned arrays are internal inputs to a later complete-source scorer,
    not partial development feedback. All required fold people and conditions
    must return; no score, margin, selection, frozen scientific method or
    first32-event/source qualification follows from this composition.
    """
    if not callable(budget_check):
        raise ValueError("explicit_allocation_owned_budget_callback_required")
    scope, baseline, references = _fit_bindings(registry, baseline_fit, reference_fits, budget_check)
    people, prepared, windows = _prepare_validation(registry, scope, validation_participants,
                                                   eeg_windows_by_condition, budget_check)
    results = {}
    for participant in people:
        budget_check()
        item = prepared[participant]
        count, start = item["prefix_count"], item["suffix_start"]
        prefix_rows, suffix_rows = item["rows"][:count], item["rows"][start:]
        prefix_keys, suffix_keys = item["keys"][:count], item["keys"][start:]
        baseline_prefix, baseline_suffix = {}, {}
        for seed, model in zip(SEEDS, baseline.models):
            if count:
                baseline_prefix[seed] = _predict_baseline(model, prefix_rows,
                    recipe=baseline.recipe, check=budget_check)
            baseline_suffix[seed] = _predict_baseline(model, suffix_rows,
                recipe=baseline.recipe, check=budget_check)
        prefix_offsets = _readonly(_mean(baseline_prefix, prefix_keys)) if count else None
        suffix_offsets = _readonly(_mean(baseline_suffix, suffix_keys))
        predictions, calibrations = {}, {}
        predictions["non_neural"], calibrations["non_neural"] = _adapt(
            baseline_prefix, baseline_suffix, item, budget_check)
        for condition in EEG_CONDITIONS:
            prefix, suffix = {}, {}
            for seed in SEEDS:
                model = references[condition].models[seed]
                if count:
                    prefix[seed] = _predict_reference(model, prefix_rows, prefix_keys,
                        windows[condition], prefix_offsets, budget_check)
                suffix[seed] = _predict_reference(model, suffix_rows, suffix_keys,
                    windows[condition], suffix_offsets, budget_check)
            predictions[condition], calibrations[condition] = _adapt(prefix, suffix, item, budget_check)
        offsets = (np.concatenate((prefix_offsets, suffix_offsets), axis=0) if count else suffix_offsets)
        results[participant] = {"row_keys": suffix_keys,
            "observed_category": _readonly(item["category"][start:]),
            "complete_bins": _readonly(item["complete"][start:]),
            "observed_through_bins": _readonly(item["coverage"][start:]),
            "predictions": predictions, "calibration_reports": calibrations,
            "evaluation_offsets": {"row_keys": item["active_keys"],
                "log_probabilities": _readonly(offsets),
                "prediction_role": "validation_rows_of_frozen_fitting_fold_baseline",
                "calibration": "uncalibrated_before_EEG_fusion"}}
    budget_check()
    return {"source": scope.source, "stage": "development_fold_validation_predictions",
        "fitting_scope": scope, "participants": people, "conditions": CONDITIONS,
        "baseline_recipe_id": baseline.recipe.recipe_id,
        "reference_recipe_id": references[EEG_CONDITIONS[0]].recipe.recipe_id,
        "completed_validation_fold": True, "complete_source_panel": False,
        "aggregate_feedback_produced": False, "encoder_fits_performed": 0,
        "source_admission_established": False, "condition_construction_established": False,
        "calibration_prefix_qualification_established": False, "participant_predictions": results}
