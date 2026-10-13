"""EEG reference ensembles over already admitted fitting arrays.

The explicit final-refit entrypoint fits three fresh seeds for ONE WAY person
on every series1–7, or on all15 self-paced development people. The separate
development entrypoint uses one existing five-series WAY or twelve-person
self-paced fitting fold. Both fit one registered recipe and supplied condition;
neither is the complete five-family/source/condition panel. The caller
freezes the recipe and supplies genuine condition-specific windows, common
training-only voltage preprocessing and the selected frozen baseline offsets.
No surrogate, mask, sensor conversion, source clock or empirical freeze is
established here. No source reader, search, calibration, scoring or CLI exists.

Numerical fitting and ensemble arithmetic are the existing backends unchanged.
Inputs remain memory-resident and must stay fixed during the call. Classical
fitting has no internal budget callback; the caller must also impose an outer
process limit. Required failures propagate without retry or a partial ensemble.
"""
from dataclasses import replace
from collections.abc import Mapping

import numpy as np
from scipy.special import logsumexp

from baseline_final_refit import FrozenFittingOffsets, _recipe as _baseline_recipe
from classical_forecast_backend import (
    ClassicalForecastModel, FittingExample, UnresolvedFit, _labels,
    canonical_reference_scope,
)
from ensemble_forecast_backend import EnsembleForecastModel, SEEDS, _row_keys
from model_recipes import reference_recipes


def _new_model(recipe):
    # Select only the requested EEG backend; do not instantiate other families.
    if recipe.family in ("past_only_filterbank_bandpower_elastic_net",
                         "past_only_covariance_riemannian_tangent_model"):
        return ClassicalForecastModel()
    if recipe.family == "small_one_directional_GRU":
        from gru_forecast_backend import GRUForecastModel
        return GRUForecastModel()
    if recipe.family == "dilated_past_only_temporal_convolutional_network":
        from tcn_forecast_backend import TCNForecastModel
        return TCNForecastModel()
    if recipe.family == "past_only_compact_EEGNet":
        from eegnet_forecast_backend import EEGNetForecastModel
        return EEGNetForecastModel()
    raise ValueError("one of the five registered EEG reference families required")


def _bound_rows(batches, fitting_offsets, registry, scope, check):
    rows = []
    for batch in batches:
        check()
        for row in batch:
            check()
            if not isinstance(row, FittingExample):
                raise ValueError("existing EEG FittingExample rows required")
            rows.append(row)
    rows = tuple(rows)
    keys = _row_keys((row.identity, row.decision_time) for row in rows)
    # All declared people/series must exist before any offset/window value or fit.
    scope.validate([row.identity for row in rows])
    packets = tuple(fitting_offsets)
    if (any(not isinstance(packet, FrozenFittingOffsets) for packet in packets)
            or len(packets) != len(scope.allowed_participants)
            or {packet.participant for packet in packets} != set(scope.allowed_participants)):
        raise UnresolvedFit("complete_final_cohort_frozen_baseline_offsets_required")
    bindings, recipes = {}, set()
    for packet in packets:
        check()
        if (packet.fitting_scope != scope
                or packet.prediction_role != "training_rows_of_frozen_baseline_not_out_of_fold"
                or packet.calibration != "uncalibrated_before_EEG_fusion"
                or packet.seed_aggregation != "mean_of_three_probability_vectors"):
            raise ValueError("unchanged_final_fitting_offset_scope_and_role_required")
        _baseline_recipe(registry, scope.source, packet.recipe_id)
        recipes.add(packet.recipe_id)
        packet_keys = _row_keys(packet.row_keys)
        if any(identity.source != scope.source or identity.participant != packet.participant
               for identity, _ in packet_keys):
            raise ValueError("baseline_offset_rows_must_belong_to_their_declared_person")
        expected = {key for key in keys if key[0].participant == packet.participant}
        if set(packet_keys) != expected:
            raise ValueError("exact_fitting_row_identity_join_required_no_extra_missing_or_duplicate_offsets")
        bindings[packet.participant] = (packet, packet_keys)
    if len(recipes) != 1:
        raise ValueError("one_common_registered_source_baseline_recipe_required")
    _labels([row.observed_category for row in rows], [row.complete_bins for row in rows], len(rows))
    offsets = {}
    for packet, packet_keys in bindings.values():
        check()
        value = packet.log_probabilities
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64")
                or value.shape != (len(packet_keys), 19) or not np.isfinite(value).all()
                or not np.allclose(logsumexp(value, axis=1), 0., atol=1e-12, rtol=0)):
            raise ValueError("frozen_float64_normalized_baseline_training_offsets_required")
        snapshot = value.copy()
        snapshot.setflags(write=False)
        for index, key in enumerate(packet_keys):
            offsets[key] = snapshot[index]
    joined = []
    for row, key in zip(rows, keys):
        check()
        supplied = np.asarray(row.context_log_probabilities)
        if supplied.shape != (19,) or not np.array_equal(supplied, offsets[key]):
            raise ValueError("EEG_row_offset_differs_from_its_frozen_baseline_join")
        # Retain the original window, label, censoring, identity and input order.
        joined.append(replace(row, context_log_probabilities=offsets[key]))
    return tuple(joined), next(iter(recipes))


def _fit_reference_ensemble(batches, *, registry, recipe_id, condition, fitting_scope,
                            fitting_offsets, semantics, budget_check, fitting_stage):
    """Shared existing fitting/offset loop; public entrypoints fix the stage."""
    if not callable(budget_check):
        raise ValueError("allocation_owned_budget_callback_required")
    budget_check()
    scope = canonical_reference_scope(registry, fitting_scope, fitting_stage=fitting_stage)
    recipes = {recipe.recipe_id: recipe for recipe in reference_recipes(registry)}
    if recipe_id not in recipes:
        raise ValueError("one_caller_frozen_registered_reference_recipe_required")
    recipe = recipes[recipe_id]
    if tuple(registry["seeds"]) != SEEDS or recipe.seeds != SEEDS:
        raise ValueError("exactly_three_unchanged_registered_seeds_required")
    if not isinstance(semantics, Mapping):
        raise ValueError("explicit_admitted_preprocessing_semantics_required")
    definition = registry["matched_conditions"]
    primary = (definition["primary"] if fitting_stage == "final_refit" else
               ("real_EEG", "zero_EEG", *definition["development_surrogates"]))
    if condition in primary:
        adapter = "original"
    elif condition in definition["factorial"]:
        adapter = "factorial"
    else:
        raise ValueError("one_registered_primary_or_factorial_condition_required")
    if semantics.get("adapter") != adapter:
        raise ValueError("condition_and_original_or_factorial_adapter_must_agree")
    for name, expected in (("registry", registry), ("fitting_scope", scope),
                           ("fitting_stage", fitting_stage)):
        if name in semantics and semantics[name] != expected:
            raise ValueError("conflicting_" + fitting_stage + "_semantics:" + name)
    if "budget_check" in semantics and semantics["budget_check"] is not budget_check:
        raise ValueError("one_caller_allocation_budget_callback_required")
    fit_semantics = dict(semantics, registry=registry, fitting_scope=scope,
                         fitting_stage=fitting_stage, budget_check=budget_check)
    rows, baseline_recipe_id = _bound_rows(batches, fitting_offsets, registry, scope, budget_check)
    members = []
    for seed in SEEDS:
        budget_check()
        model = _new_model(recipe)
        report = model.fit((rows,), recipe=recipe, seed=seed, semantics=fit_semantics)
        budget_check()
        if (not isinstance(report, Mapping) or getattr(model, "fitted", False) is not True
                or not (report.get("completed") is True or report.get("converged") is True)):
            raise UnresolvedFit("required_final_reference_seed_did_not_complete")
        members.append((model, report))
    ensemble = EnsembleForecastModel(members, registry=registry, fitting_scope=scope,
                                     fitting_stage=fitting_stage)
    budget_check()
    return {"source": scope.source, "stage": ("final_reference_refit" if fitting_stage == "final_refit"
                                               else "development_reference_fitting_fold"),
        "completed_scope": True, "complete_reference_panel": False,
        "selected_recipe_id": recipe.recipe_id, "condition": condition,
        "fitting_scope": scope, "seeds": SEEDS, "fits_completed": len(members),
        "fit_reports": tuple(report for _, report in members), "ensemble": ensemble,
        "baseline_recipe_id": baseline_recipe_id,
        "offsets_are_out_of_fold": False, "calibration_performed": False,
        "final_evaluation_performed": False, "source_admission_established": False,
        "empirical_method_freeze_established": False,
        "condition_construction_established": False,
        "production_resource_qualification_established": False}


def fit_final_reference_ensemble(batches, *, registry, selected_recipe_id, condition,
                                fitting_scope, fitting_offsets, semantics, budget_check):
    """Refit exactly three seeds for one complete final fitting scope.

    ``selected_recipe_id`` is caller-frozen, not selected from scores here.
    ``fitting_offsets`` are existing FrozenFittingOffsets from the same scope;
    each person's row order may differ, but its identity set must match exactly.
    Existing example offsets must agree, preventing a silent offset replacement.
    The condition name binds the compatible original/factorial adapter only;
    authentic input transformation and source access remain caller obligations.
    """
    return _fit_reference_ensemble(batches, registry=registry, recipe_id=selected_recipe_id,
        condition=condition, fitting_scope=fitting_scope, fitting_offsets=fitting_offsets,
        semantics=semantics, budget_check=budget_check, fitting_stage="final_refit")


def fit_development_reference_ensemble(batches, *, registry, recipe_id, condition,
                                      fitting_scope, fitting_offsets, semantics, budget_check):
    """Fit one registered condition in ONE canonical development fitting fold.

    Offsets come from the already frozen source-specific baseline fitted only
    on this same fold. They remain training predictions, never OOF evidence.
    All five WAY fitting series or all twelve self-paced fitting people are
    required. The validation series/people, WAY feedback series7 and final
    fitting scopes are rejected before fitting. Only the two registered
    development surrogates, real/zero EEG and existing factorial conditions
    are accepted; this call cannot enlarge the reference configuration menu.
    Factorial conditions require the caller's existing development-frozen
    one-per-family recipe; this entrypoint does not establish that selection.

    The caller supplies the genuine condition-specific input and fitting-only
    preprocessing. This is a fitted object for subsequent admitted validation,
    not partial fold feedback, recipe selection, calibration or panel scoring.
    Failures propagate without retry or a smaller successful ensemble.
    """
    return _fit_reference_ensemble(batches, registry=registry, recipe_id=recipe_id,
        condition=condition, fitting_scope=fitting_scope, fitting_offsets=fitting_offsets,
        semantics=semantics, budget_check=budget_check, fitting_stage="development")
