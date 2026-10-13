"""EP19 fixed three-seed probability ensemble and calibration integration.

No source I/O, fitting of encoders, prefix selection or held-out admission occurs.
The caller supplies one matched condition's already-fitted seed models. Saved
prediction rows must align exactly with the admitted natural-prevalence prefix;
its chronological first-32-event selection, label truncation, history guard and
eligibility remain upstream responsibilities. This module cannot certify them.
"""
from numbers import Real
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from classical_forecast_backend import (
    FittingScope, GroupIdentity, UnresolvedFit, _labels, exact_samples,
    ensemble_log_probabilities, canonical_reference_scope,
)
from forecast_components import calibrated_log_probabilities, fit_four_parameter_calibration
from model_recipes import reference_recipes

SEEDS = (1901, 1902, 1903)


def _seed_arrays(seed_log_probabilities, rows):
    if not isinstance(seed_log_probabilities, Mapping):
        raise ValueError("explicit seed-to-prediction mapping required")
    if any(type(seed) is not int or seed not in SEEDS for seed in seed_log_probabilities):
        raise ValueError("only the three registered integer seed identities are permitted")
    if set(seed_log_probabilities) != set(SEEDS):
        raise UnresolvedFit("all_three_seed_predictions_required_no_partial_ensemble")
    arrays = []
    for seed in SEEDS:
        value = seed_log_probabilities[seed]
        if np.iscomplexobj(value):
            raise ValueError("seed log probabilities must be real")
        array = np.asarray(value, dtype=np.float64)
        if (array.shape != (rows, 19) or not np.all(np.isfinite(array))
                or not np.allclose(logsumexp(array, axis=1), 0., rtol=0, atol=1e-12)):
            raise UnresolvedFit("each_seed_requires_complete_finite_normalized_19_category_predictions")
        arrays.append(array)
    return arrays


def _mean_seed_probabilities(seed_log_probabilities, rows):
    # Arithmetic mean in probability space, evaluated without exponentiating
    # very small probabilities. Never mean logits/log probabilities or pick a seed.
    return ensemble_log_probabilities(np.stack(_seed_arrays(seed_log_probabilities, rows)))


def _row_keys(keys):
    keys = tuple(keys)
    if not keys:
        raise ValueError("nonempty admitted prediction-row order required")
    for row in keys:
        if (not isinstance(row, tuple) or len(row) != 2 or not isinstance(row[0], GroupIdentity)
                or row[0].source not in ("WAY", "self_paced") or row[0].participant is None
                or row[0].series_or_run is None
                or not isinstance(row[0].boundary_id, str) or not row[0].boundary_id
                or not isinstance(row[1], Real) or not np.isfinite(row[1])):
            raise ValueError("row keys must be existing GroupIdentity and finite decision time pairs")
    if len(set(keys)) != len(keys):
        raise ValueError("prediction row identities may not be duplicated")
    return keys


def mean_seed_log_probabilities(seed_log_probabilities, *, seed_row_keys, expected_row_keys):
    """Combine only exact matching ordered rows from all three fixed seeds."""
    expected = _row_keys(expected_row_keys)
    if not isinstance(seed_row_keys, Mapping) or set(seed_row_keys) != set(SEEDS):
        raise UnresolvedFit("all_three_seed_row_orders_required")
    for seed in SEEDS:
        if _row_keys(seed_row_keys[seed]) != expected:
            raise ValueError("seed prediction order differs from the admitted row order")
    return _mean_seed_probabilities(seed_log_probabilities, len(expected))


def fit_ensemble_calibration(seed_log_probabilities, *, seed_row_keys, prefix_row_keys,
                             observed_category, complete_bins, weights):
    """Fit one model/person's four parameters AFTER the fixed probability mean.

    Input arrays contain only the caller-selected whole chronological prefix,
    including natural at-risk stillness and labels censored at its endpoint.
    No guard/suffix data or model objects are accepted, and no encoder, scaler,
    input state or per-bin bias can be changed by this operation. The existing
    optimizer and flagged identity-fallback rule are reused unchanged.
    """
    rows = _row_keys(prefix_row_keys)
    groups = {(identity.source, identity.participant) for identity, _ in rows}
    if len(groups) != 1 or next(iter(groups))[0] != "self_paced":
        raise ValueError("one self-paced participant's admitted calibration prefix required")
    ensemble = mean_seed_log_probabilities(seed_log_probabilities,
        seed_row_keys=seed_row_keys, expected_row_keys=rows)
    categories, complete = _labels(observed_category, complete_bins, len(rows))
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(rows),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("aligned positive natural-prevalence prefix weights required")
    report = fit_four_parameter_calibration(ensemble, categories, complete, weights)
    return {**report, "seed_aggregation": "equal_probability_mean_before_calibration",
            "seeds": list(SEEDS), "prefix_rows": len(rows),
            "prefix_eligibility_and_timing_established_by_backend": False}


class EnsembleForecastModel:
    """A small adapter over three existing fitted EEG reference backends.

    `members` are (backend, returned_fit_report) pairs for ONE matched EEG
    condition. Actual model scopes and existing reports must agree with the
    supplied fitting scope. No report is a source-access grant. Call reset at
    each boundary and before suffix inference.
    """
    def __init__(self, members, *, registry, fitting_scope, fitting_stage="development"):
        self.fitted = False
        if not isinstance(fitting_scope, FittingScope):
            raise ValueError("existing canonical development FittingScope required")
        scope = canonical_reference_scope(registry, fitting_scope, fitting_stage=fitting_stage)
        members = tuple(members)
        if len(members) != 3:
            raise UnresolvedFit("all_three_fitted_seed_models_required")
        if tuple(registry["seeds"]) != SEEDS:
            raise ValueError("unchanged three-seed bank required")
        registered = {recipe.recipe_id: recipe for recipe in reference_recipes(registry)}
        self.models = {}
        common = None
        for model, report in members:
            if not isinstance(report, Mapping):
                raise ValueError("each backend's existing fit report is required")
            if (getattr(model, "fitted", False) is not True
                    or not (report.get("completed") is True or report.get("converged") is True)):
                raise UnresolvedFit("incomplete_seed_fit_cannot_enter_ensemble")
            seed, recipe = getattr(model, "seed", None), getattr(model, "recipe", None)
            if type(seed) is not int or seed not in SEEDS or seed in self.models or report.get("seed") != seed:
                raise UnresolvedFit("exactly_one_fitted_model_for_each_registered_seed_required")
            if (recipe is None or recipe.recipe_id not in registered or recipe != registered[recipe.recipe_id]
                    or report.get("recipe_id") != recipe.recipe_id or report.get("fold_id") != scope.fold_id
                    or report.get("source", scope.source) != scope.source
                    or getattr(model, "fitting_scope", None) != scope):
                raise ValueError("seed recipe and fitting fold must match existing registered metadata")
            if (getattr(model, "fitting_stage", "development") != fitting_stage
                    or report.get("fitting_stage", "development") != fitting_stage):
                raise ValueError("seed model and report must match the explicit fitting stage")
            rate, adapter = getattr(model, "rate", None), getattr(model, "adapter", None)
            exact_samples(recipe.history_seconds, rate)
            exact_samples(.2, rate)
            if adapter not in ("original", "factorial"):
                raise ValueError("explicit fitted original/factorial adapter required")
            filters = None
            if recipe.family == "past_only_filterbank_bandpower_elastic_net":
                coefficients = getattr(getattr(model, "features", None), "coefficients", None)
                if coefficients is None or len(coefficients) != 5:
                    raise ValueError("fitted classical seed must retain its five causal FIRs")
                filters = tuple(tuple(np.asarray(value, dtype=np.float64)) for value in coefficients)
            signature = (recipe, rate, adapter, filters)
            if common is not None and signature != common:
                raise ValueError("all seeds must share recipe, analysis grid, adapter and causal FIRs")
            common = signature
            self.models[seed] = model
        if len({id(model) for model in self.models.values()}) != 3:
            raise ValueError("each seed must retain its separate fitted model")
        self.recipe, self.rate, self.adapter = common[:3]
        self.fitting_scope = scope
        self.fitting_stage = fitting_stage
        self.boundary_id, self.last_decision_time = None, -np.inf
        self.fitted = True

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("explicit nonempty reset boundary required")
        self.boundary_id = None
        for seed in SEEDS:
            self.models[seed].reset(boundary_id=boundary_id)
        self.boundary_id, self.last_decision_time = boundary_id, -np.inf

    def predict_log_probabilities(self, past_input, *, decision_time, context_log_probabilities):
        if self.boundary_id is None or not np.isfinite(decision_time) or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        try:
            predictions = {seed: np.asarray(self.models[seed].predict_log_probabilities(past_input,
                decision_time=decision_time, context_log_probabilities=context_log_probabilities))[None, :]
                for seed in SEEDS}
            result = _mean_seed_probabilities(predictions, 1)[0]
        except Exception:
            # A prior seed may already have advanced its interface clock. No
            # partial output or implicit retry can proceed without explicit reset.
            self.boundary_id = None
            raise
        self.last_decision_time = decision_time
        return result

    def predict_paired_log_probabilities(self, past_input, *, decision_time,
                                        context_log_probabilities, calibration_parameters):
        """Return both suffix conditions from the SAME encoder-state evaluation.

        Parameters must come from this model/person's separately fitted prefix.
        This method takes no labels and never advances the encoder twice to
        obtain the no-label and calibrated versions of the same decision.
        """
        # Validate before advancing any model state.
        calibrated_log_probabilities(np.full((1, 19), -np.log(19.)), calibration_parameters)
        zero_label = self.predict_log_probabilities(past_input, decision_time=decision_time,
            context_log_probabilities=context_log_probabilities)
        calibrated = calibrated_log_probabilities(zero_label[None], calibration_parameters)[0]
        return {"zero_label": zero_label, "calibrated": calibrated}
