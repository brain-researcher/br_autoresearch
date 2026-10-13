"""Role-bound primary EP07 fit and development prediction assembly.

One caller-selected recipe is applied across all animal/day/region settings;
parameters are fitted separately. This module has no scorer, outcome reader,
hyperparameter search, file access or held-out-opening interface. Source masses
are explicit candidate parameters, distinct from feature-fit day weights and
from the animal-balanced scientific score. Learned source weighting, set
encoders, ensembles and alternative-explanation interfaces remain separate
extensions of the registered grammar, not excluded candidate families.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np

from lfp_features import (FeatureRecipe, fit_lfp_features, transform_lfp_features,
                          transform_lfp_held_out)
from native_trials import DayTrials
from primary_fit_scope import bind_fit_days, validate_control
from primary_design import (fit_primary_design, make_primary_fit_design,
    transform_primary_features, restore_primary_prediction,
    transform_primary_held_out_features, restore_primary_held_out_prediction)
from primary_population import (fit_primary_population, project_primary_responses,
    reconstruct_primary_responses)
from regression import RegressionRecipe, fit_regression, predict_regression


@dataclass(frozen=True)
class PrimaryRecipe:
    features: FeatureRecipe
    population_rank: int
    alignment: str
    alignment_regularization: float | None
    adjustment: str
    regression: RegressionRecipe
    calibration_mass: float = 1.0
    source_masses: tuple[float, ...] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class FittedPrimary:
    recipe: PrimaryRecipe
    features: object
    population: object
    design: object
    regression: object
    day_masses: tuple[tuple[str, float], ...]
    rows_per_day: tuple[tuple[str, int], ...]


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned resource check required")
    check()


def _masses(recipe, days):
    if (not isinstance(recipe.source_masses, tuple)
            or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   or not np.isfinite(value) or value <= 0
                   for value in recipe.source_masses + (recipe.calibration_mass,))):
        raise ValueError("explicit finite positive day masses required")
    if days[-1].stream == "today_only":
        if recipe.source_masses:
            raise ValueError("today-only recipe must contain no source masses")
        return (1.0,)
    if len(recipe.source_masses) != len(days) - 1:
        raise ValueError("history masses must bind every declared chronological source day")
    values = np.asarray(recipe.source_masses + (recipe.calibration_mass,), dtype=np.float64)
    # Scaling first avoids overflow when normalizing large finite masses.
    values /= np.max(values)
    values /= np.sum(values)
    return tuple(float(value) for value in values)


def fit_primary_model(days, *, recipe, check):
    """Fit from exact complete candidate fit packs; never development arrays."""
    return _fit_primary_model(days, recipe=recipe, check=check)


def _validate_recipe(recipe, *, _control=None):
    """Keep ordinary grammar strict, including when called by a control wrapper."""
    if not isinstance(recipe, PrimaryRecipe):
        raise TypeError("explicit primary candidate recipe required")
    if recipe.adjustment not in ("explicit", "residualize", "both") and not (
            recipe.adjustment == "none" and _control is not None
            and validate_control(_control).kind == "remove_nuisance"):
        raise ValueError("unknown primary direction/time adjustment")
    if (not isinstance(recipe.regression, RegressionRecipe)
            or (recipe.adjustment in ("explicit", "both")
                and recipe.regression.fit_intercept is not True)):
        raise ValueError("the drop-first direction/time design requires a fitted intercept")


def _fit_primary_model(days, *, recipe, check, _control=None):
    """Shared numerical assembly; only primary_control supplies a control scope."""
    _check(check)
    _validate_recipe(recipe, _control=_control)
    days = bind_fit_days(days, _control=_control)
    masses = _masses(recipe, days)
    context = {} if _control is None else {"_control": _control}
    # Bases/maps are fitted to unresidualized counts before any residual design.
    population = fit_primary_population(days, rank=recipe.population_rank,
        alignment=recipe.alignment, regularization=recipe.alignment_regularization,
        check=check, **context)
    features = fit_lfp_features(days, recipe=recipe.features, check=check, **context)
    fit_features = tuple(transform_lfp_features(features, day, check=check) for day in days)
    design = fit_primary_design(days, fit_features, adjustment=recipe.adjustment,
                                check=check, **context)
    xs, ys, weights, rows_per_day = [], [], [], []
    for day, feature, mass in zip(days, fit_features, masses):
        _check(check)
        batch = make_primary_fit_design(design, day, feature, check=check)
        response = project_primary_responses(population, day, batch.responses,
            residualized=recipe.adjustment in ("residualize", "both"), check=check)
        x = batch.design.X
        y = response.reshape(-1, recipe.population_rank)
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(x), mass / len(x), dtype=np.float64))
        rows_per_day.append((day.day, len(x)))
    _check(check)
    regression = fit_regression(np.concatenate(xs), np.concatenate(ys),
        row_weights=np.concatenate(weights), recipe=recipe.regression, check=check)
    _check(check)
    return FittedPrimary(recipe, features, population, design, regression,
        tuple((day.day, mass) for day, mass in zip(days, masses)), tuple(rows_per_day))


def predict_primary_development(fitted, day, *, check):
    """Use frozen fitting objects and features only; return native predictions.

    Final-opening and falsifier roles are not accepted by this interface.
    No direction, trial, bin or neuron is discarded to improve a score.
    """
    return _predict_primary_features(fitted, day, check=check)


def predict_primary_held_out(fitted, day, *, check):
    """Transport a frozen fit onto explicit target held-out FEATURES only.

    Fitted parameters come from the original training/calibration fit. This
    wrapper performs no fitting, selection, neural outcome read or opening
    authorization. Weighted candidates pass their already fitted .primary;
    ensembles predict frozen members separately before fixed-weight composition.
    """
    return _predict_primary_features(fitted, day, check=check, _phase="held_out")


def _predict_primary_features(fitted, day, *, check, _phase="candidate"):
    _check(check)
    if not isinstance(fitted, FittedPrimary):
        raise TypeError("fitted primary candidate required")
    expected = (("development_features", "development") if _phase == "candidate" else
                ("held_out_features", "held_out") if _phase == "held_out" else None)
    if (expected is None or not isinstance(day, DayTrials) or (day.view_role, day.role) != expected
            or day.spikes is not None
            or day.unit_guide is not None):
        raise PermissionError("prediction requires the explicit feature-only phase")
    transform, design_transform, restore = (
        (transform_lfp_features, transform_primary_features, restore_primary_prediction)
        if _phase == "candidate" else
        (transform_lfp_held_out, transform_primary_held_out_features, restore_primary_held_out_prediction))
    features = transform(fitted.features, day, check=check)
    design = design_transform(fitted.design, features, check=check)
    coordinates = predict_regression(fitted.regression, design.X, check=check)
    coordinates = coordinates.reshape(len(day.trial_ids), len(day.bin_offsets),
                                      fitted.recipe.population_rank)
    native = reconstruct_primary_responses(fitted.population, coordinates,
        residualized=fitted.recipe.adjustment in ("residualize", "both"), check=check)
    result = restore(fitted.design, design, native, check=check)
    _check(check)
    return result
