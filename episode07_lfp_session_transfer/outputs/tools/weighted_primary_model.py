"""Dormant history-only source-weight fitting assembly for EP07 primary models.

Each call fits weights afresh from exactly the three earlier training packs
and the same target calibration pack used by its primary model. No fitted
weights, features or primary object can be supplied to fitting, and today-only
packs are refused. This wrapper does not read files, choose a recipe, score
development responses, or implement the conditional residual follow-up.

The base PrimaryRecipe must have neutral source_masses=(1,1,1) and
calibration_mass=1. SourceWeightRecipe alone owns the effective masses; they
replace, rather than multiply, those defaults. Feature-transform fitting still
uses its registered equal-day weighting, separately from regression day masses
and from animal-balanced scientific scoring weights.

Calibration similarity first fits the base feature recipe on the exact history
fit roster for its weight calculation. The primary fitter then independently
refits that same deterministic recipe on those same packs. This redundant
computation avoids importing fitted state into the primary fitter; it does not
change fitting data, numerical feature conventions or scientific opportunities.
Other weighting families do not require this extra feature fit. Prediction
uses only the frozen primary model, never recalculating weights from new LFP.

Reliability and hierarchical shrinkage retain source_weighting's explicit
mean-trajectory heuristic interpretation. Leave-one-day stacking and trainable
set encoders require separate interfaces and are not implemented here.
"""
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from lfp_features import fit_lfp_features
from primary_model import (FittedPrimary, PrimaryRecipe, fit_primary_model,
                           predict_primary_development, _fit_primary_model, _validate_recipe)
from source_weighting import (SourceWeights, fit_source_weights,
                              _recipe as _weight_recipe, _roster)


@dataclass(frozen=True)
class FittedWeightedPrimary:
    base_recipe: PrimaryRecipe
    source_weights: SourceWeights
    effective_recipe: PrimaryRecipe
    primary: FittedPrimary


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _neutral_masses(recipe):
    if not isinstance(recipe, PrimaryRecipe):
        raise TypeError("an unfitted PrimaryRecipe is required")
    if not isinstance(recipe.source_masses, tuple) or len(recipe.source_masses) != 3:
        raise ValueError("base source masses must be the neutral tuple (1,1,1)")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or value != 1
           for value in recipe.source_masses + (recipe.calibration_mass,)):
        raise ValueError("source-weight recipe exclusively owns masses; base masses must all equal one")


def fit_weighted_primary_model(days, *, recipe, weight_recipe, check):
    """Fit one history candidate and retain exact weights plus effective recipe.

    Canonicalize the complete day roster before either fitting layer runs, so
    source masses always bind chronological source-day identities. Both role
    validation and conflicting-mass refusal precede interpreting value arrays.
    Immutable adapter packs are required by the surrounding candidate workflow;
    metadata checks cannot authenticate arbitrary caller-constructed arrays.
    """
    return _fit_weighted_primary_model(days, recipe=recipe, weight_recipe=weight_recipe, check=check)


def _fit_weighted_primary_model(days, *, recipe, weight_recipe, check, _control=None):
    """Shared fit for the ordinary roster or an explicitly selected control."""
    _check(check)
    _neutral_masses(recipe)
    _validate_recipe(recipe, _control=_control)
    weight_recipe = _weight_recipe(weight_recipe)
    days = _roster(days, _control=_control)
    context = {} if _control is None else {"_control": _control}
    _check(check)
    weight_features = (fit_lfp_features(days, recipe=recipe.features, check=check, **context)
                       if weight_recipe.family == "calibration_similarity" else None)
    weights = fit_source_weights(days, recipe=weight_recipe,
                                 feature_model=weight_features, check=check, **context)
    effective = replace(recipe, source_masses=weights.source_masses,
                        calibration_mass=weights.target_mass)
    _check(check)
    primary = (fit_primary_model(days, recipe=effective, check=check) if _control is None else
               _fit_primary_model(days, recipe=effective, check=check, _control=_control))
    _check(check)
    return FittedWeightedPrimary(recipe, weights, effective, primary)


def predict_weighted_primary_development(fitted, day, *, check):
    """Delegate feature-only prediction to the frozen history primary object."""
    _check(check)
    if not isinstance(fitted, FittedWeightedPrimary) or not isinstance(fitted.primary, FittedPrimary):
        raise TypeError("one fitted weighted primary candidate is required")
    if (fitted.source_weights.stream != "history_assisted"
            or fitted.primary.features.stream != "history_assisted"):
        raise PermissionError("weighted primary prediction cannot reuse today-only fitted state")
    return predict_primary_development(fitted.primary, day, check=check)
