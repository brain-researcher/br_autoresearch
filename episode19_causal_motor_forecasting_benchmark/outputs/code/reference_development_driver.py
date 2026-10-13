"""One fixed EP19 EEG recipe through every admitted development fold.

This array-only composition reuses caller-owned, already fitted baselines.
The caller supplies authentic condition windows, fitting-only preprocessing,
qualified labels/prefixes and the independent complete scored-row roster.
It constructs no condition, opens no source, selects no recipe and grants no
empirical access. All four conditions use the same fitting rows and labels.

All inputs and fold prediction arrays remain resident. EEG models are retained
for only one fold and released after its internal predictions. Classical fits
cannot be interrupted by the callback alone; the caller must enforce an outer
process/allocation limit. Production memory and runtime remain unmeasured.
Failures propagate without retry, partial-source feedback or smaller cohorts.
"""
from collections.abc import Mapping

import numpy as np

from baseline_final_refit import FittedBaselineEnsemble, FrozenFittingOffsets
from classical_forecast_backend import FittingExample, PastWindow, UnresolvedFit, _labels
from development_baseline_driver import PreparedBaselineRow, PreparedParticipant, baseline_recipes
from ensemble_forecast_backend import SEEDS, _row_keys
from model_recipes import reference_recipes
from reference_development_panel import _expected_rows, score_reference_development_panel
from reference_development_prediction import EEG_CONDITIONS, _prepare_validation, predict_reference_development_fold
from reference_final_refit import _bound_rows, fit_development_reference_ensemble


def _baseline(result, registry, scope, recipe_id, check):
    """Reject an incomplete or differently scoped supplied baseline before fitting."""
    if (not isinstance(result, Mapping) or result.get("completed") is not True
            or result.get("stage") != "development_baseline_inside_EEG_fitting_fold"
            or result.get("source") != scope.source or result.get("selected_recipe_id") != recipe_id
            or result.get("fits_completed") != 3 or tuple(result.get("seeds", ())) != SEEDS):
        raise UnresolvedFit("complete_selected_fitting_fold_baseline_required")
    ensemble = result.get("ensemble")
    recipes = {recipe.recipe_id: recipe for recipe in baseline_recipes(registry, source=scope.source)}
    if (not isinstance(ensemble, FittedBaselineEnsemble) or ensemble.fitting_scope != scope
            or ensemble.recipe != recipes[recipe_id] or len(ensemble.models) != 3
            or len(ensemble.fit_reports) != 3 or len({id(model) for model in ensemble.models}) != 3):
        raise ValueError("exact_registered_baseline_scope_and_three_fitted_models_required")
    for seed, model, report in zip(SEEDS, ensemble.models, ensemble.fit_reports):
        check()
        if (not isinstance(report, Mapping) or getattr(model, "fitted", False) is not True
                or not (report.get("completed") is True or report.get("converged") is True)
                or getattr(model, "seed", None) != seed or report.get("seed") != seed
                or getattr(model, "recipe", None) != ensemble.recipe
                or report.get("recipe_id") != recipe_id
                or getattr(model, "fitting_scope", None) != scope
                or report.get("fold_id") != scope.fold_id):
            raise UnresolvedFit("all_exact_completed_baseline_seed_models_and_reports_required")


def _examples(rows, windows, offsets):
    return tuple(FittingExample(windows[row.identity, row.decision_time], row.decision_time,
        row.identity, row.observed_category, row.complete_bins,
        offsets[row.identity, row.decision_time]) for row in rows)


def _fitting_inputs(item, baseline, registry, scope, recipe_id, check):
    people = tuple(item["fitting_participants"])
    if (any(not isinstance(person, PreparedParticipant) for person in people)
            or len(people) != len(scope.allowed_participants)
            or {person.participant for person in people} != set(scope.allowed_participants)):
        raise UnresolvedFit("complete_canonical_fitting_people_required")
    by_person = {person.participant: person for person in people}
    rows = []
    for person in scope.allowed_participants:
        check()
        local = tuple(by_person[person].rows)
        if (not local or any(not isinstance(row, PreparedBaselineRow)
                            or row.identity.participant != person for row in local)):
            raise ValueError("common_prepared_fitting_rows_must_belong_to_each_declared_person")
        rows.extend(local)
    rows = tuple(rows)
    keys = _row_keys((row.identity, row.decision_time) for row in rows)
    scope.validate([row.identity for row in rows])
    _labels([row.observed_category for row in rows], [row.complete_bins for row in rows], len(rows))
    windows = item["fitting_windows_by_condition"]
    if not isinstance(windows, Mapping) or set(windows) != set(EEG_CONDITIONS):
        raise UnresolvedFit("all_four_genuine_development_conditions_required")
    for condition in EEG_CONDITIONS:
        check()
        if (not isinstance(windows[condition], Mapping) or set(windows[condition]) != set(keys)
                or any(not isinstance(window, PastWindow) for window in windows[condition].values())):
            raise ValueError("each_condition_requires_exact_common_fitting_window_identities")
    packets = tuple(baseline.get("fitting_offsets", ()))
    offsets = {}
    for packet in packets:
        check()
        if not isinstance(packet, FrozenFittingOffsets) or packet.recipe_id != recipe_id:
            raise ValueError("offsets_from_the_same_selected_baseline_required")
        packet_keys = _row_keys(packet.row_keys)
        if (not isinstance(packet.log_probabilities, np.ndarray)
                or packet.log_probabilities.shape != (len(packet_keys), 19)):
            raise ValueError("complete_19_category_baseline_offset_rows_required")
        for index, key in enumerate(packet_keys):
            if key in offsets:
                raise ValueError("duplicate_baseline_offset_identity")
            offsets[key] = packet.log_probabilities[index]
    if set(offsets) != set(keys):
        raise ValueError("baseline_offsets_must_match_every_common_fitting_row")
    # Reuse the existing packet role/scope/normalization checks before ANY fit.
    # Only one condition is needed: all other conditions share these exact
    # rows, labels and offsets. The temporary immutable copies are discarded.
    _bound_rows((_examples(rows, windows[EEG_CONDITIONS[0]], offsets),), packets,
                registry, scope, check)
    return rows, windows, packets, offsets


def _prepare_fold(item, registry, scope, baseline_id, keys, category, complete,
                  coverage, positions, check):
    if not isinstance(item, Mapping):
        raise ValueError("explicit_supplied_fold_inputs_required")
    baseline = item["baseline_fit"]
    _baseline(baseline, registry, scope, baseline_id, check)
    rows, windows, packets, offsets = _fitting_inputs(item, baseline, registry, scope, baseline_id, check)
    semantics = item["semantics"]
    if not isinstance(semantics, Mapping) or semantics.get("adapter") != "original":
        raise ValueError("explicit_original_reference_preprocessing_semantics_required")
    for name, expected in (("registry", registry), ("fitting_scope", scope), ("fitting_stage", "development")):
        if name in semantics and semantics[name] != expected:
            raise ValueError("conflicting_development_semantics:" + name)
    if "budget_check" in semantics and semantics["budget_check"] is not check:
        raise ValueError("one_allocation_owned_budget_callback_required")
    participants = tuple(item["validation_participants"])
    validation_windows = item["validation_windows_by_condition"]
    people, prepared, validated_windows = _prepare_validation(registry, scope,
        participants, validation_windows, check)
    for person in people:
        check()
        local = prepared[person]
        start = local["suffix_start"]
        indices = np.asarray(positions[scope.fold_id, person], dtype=np.intp)
        if tuple(local["keys"][start:]) != tuple(keys[index] for index in indices):
            raise ValueError("validation_suffix_must_match_independent_complete_scored_roster")
        if (not np.array_equal(local["category"][start:], category[indices])
                or not np.array_equal(local["complete"][start:], complete[indices])
                or not np.array_equal(local["coverage"][start:], coverage[indices])):
            raise ValueError("validation_labels_must_match_independent_complete_scored_roster")
    return (baseline, rows, windows, packets, offsets, dict(semantics),
            participants, validated_windows)


def _run_fold(prepared, *, registry, scope, reference_id, check):
    baseline, rows, windows, packets, offsets, semantics, participants, validation_windows = prepared
    fitted = {}
    for condition in EEG_CONDITIONS:
        check()
        fitted[condition] = fit_development_reference_ensemble(
            (_examples(rows, windows[condition], offsets),), registry=registry,
            recipe_id=reference_id, condition=condition, fitting_scope=scope,
            fitting_offsets=packets, semantics=semantics, budget_check=check)
    result = predict_reference_development_fold(registry=registry, baseline_fit=baseline,
        reference_fits=fitted, validation_participants=participants,
        eeg_windows_by_condition=validation_windows, budget_check=check)
    reports = {condition: fitted[condition]["fit_reports"] for condition in EEG_CONDITIONS}
    # Neither return value retains the twelve fitted EEG models. Returning from
    # this helper drops this fold's models before the next fold starts fitting.
    return result, reports


def run_reference_development_recipe(*, registry, source, baseline_recipe_id,
        reference_recipe_id, fold_inputs, expected_row_keys, observed_category,
        complete_bins, observed_through_bins, budget_check):
    """Execute ONE fixed recipe on all 72 WAY or five self-paced folds.

    ``fold_inputs`` maps each canonical fold ID to ``baseline_fit`` (the
    existing fit_fitting_fold_baseline result), ``fitting_participants``
    (its original PreparedParticipant inputs), ``fitting_windows_by_condition``,
    common ``semantics``, ``validation_participants``, and
    ``validation_windows_by_condition``. Each window map is keyed by the exact
    (GroupIdentity, decision_time) pairs. Validation windows omit guard rows.
    The caller must keep every supplied object fixed throughout this call.

    All source folds, conditions, supplied baseline packets and independent
    scored suffix labels bind before fitting. Qualified source facts and the
    baseline fit's original label correctness remain caller responsibilities.
    No intermediate fold score, recipe choice, new baseline fit or empirical
    permission is returned. Existing numerical backends and scorer are reused.
    """
    if not callable(budget_check):
        raise ValueError("explicit_allocation_owned_budget_callback_required")
    budget_check()
    _, roster, keys, category, complete, coverage, positions = _expected_rows(
        registry, source, expected_row_keys, observed_category, complete_bins,
        observed_through_bins, budget_check)
    if (tuple(registry["seeds"]) != SEEDS
            or tuple(registry["matched_conditions"]["development_surrogates"]) != EEG_CONDITIONS[2:]):
        raise ValueError("unchanged_development_conditions_and_three_seeds_required")
    if baseline_recipe_id not in {recipe.recipe_id for recipe in baseline_recipes(registry, source=source)}:
        raise ValueError("registered_source_baseline_recipe_required")
    if reference_recipe_id not in {recipe.recipe_id for recipe in reference_recipes(registry)}:
        raise ValueError("registered_reference_recipe_required")
    if not isinstance(fold_inputs, Mapping) or set(fold_inputs) != set(roster):
        raise UnresolvedFit("exact_complete_development_fold_roster_required")
    prepared = {}
    for fold_id, (scope, _, _) in roster.items():
        budget_check()
        prepared[fold_id] = _prepare_fold(fold_inputs[fold_id], registry, scope,
            baseline_recipe_id, keys, category, complete, coverage, positions, budget_check)
    predictions, reports = [], []
    for fold_id, (scope, _, _) in roster.items():
        budget_check()
        prediction, fits = _run_fold(prepared.pop(fold_id), registry=registry,
            scope=scope, reference_id=reference_recipe_id, check=budget_check)
        predictions.append(prediction)
        reports.append({"fold_id": fold_id, "condition_fit_reports": fits})
    budget_check()
    result = score_reference_development_panel(predictions, registry=registry, source=source,
        baseline_recipe_id=baseline_recipe_id, reference_recipe_id=reference_recipe_id,
        expected_row_keys=keys, observed_category=category, complete_bins=complete,
        observed_through_bins=coverage, budget_check=budget_check)
    result["execution"] = {"fold_order": tuple(roster), "seeds": SEEDS,
        "EEG_fits_completed": len(roster) * len(EEG_CONDITIONS) * len(SEEDS),
        "baseline_fits_performed": 0, "condition_fit_reports": tuple(reports),
        "condition_construction_established": False,
        "production_resource_qualification_established": False}
    budget_check()
    return result
