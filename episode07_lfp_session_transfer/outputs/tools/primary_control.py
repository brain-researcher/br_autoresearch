"""Explicit fitting-only EP07 source-removal and nuisance-removal controls.

Supply the complete original training/calibration roster and the unfitted,
frozen candidate recipe. Metadata is bound before selection; only retained
packs reach feature, population, design, source-weight and regression fitting.
There are no readers, scorer, search, fitted-state inputs or held-out roles.
Predictions use the existing feature-only development interface unchanged.

Omitting a source preserves the total source mass and calibration mass, and
fixed masses retain their relative shares. Learned weights are recomputed on
the remaining sources under the same weight recipe and total masses. Feature
fit weighting remains equal across retained days, independently of regression
masses. Similarity uses the wrapper's redundant deterministic feature refit on
the same retained packs, not an inherited transform.

Removing all history fits the HISTORY recipe anew from calibration alone:
its feature family/scaling, explicit population rank, regression and adjustment
are retained. The declared alignment has zero source maps. There are no fitted
source weights; calibration has all row mass. This is not the separately tuned
today-only model. Removing nuisance alone permits adjustment='none', with raw
count PCA centering/restoration and no fitted cell means or cell predictors.

The control tag records an intervention, not success of a scientific check or
authority for an outcome opening. Ensemble members must each be refitted by
their caller with unchanged membership and mixing weights.
"""
from dataclasses import dataclass, replace
import math

from primary_fit_scope import PrimaryControl, bind_fit_days, source_days, validate_control
from primary_model import (FittedPrimary, PrimaryRecipe, _check, _fit_primary_model,
                           _masses, _validate_recipe, predict_primary_development)
from source_weighting import SourceWeights, SourceWeightRecipe, _recipe as _weight_recipe
from weighted_primary_model import _fit_weighted_primary_model, _neutral_masses


@dataclass(frozen=True)
class FittedPrimaryControl:
    control: PrimaryControl
    original_recipe: PrimaryRecipe
    weight_recipe: SourceWeightRecipe | None
    effective_recipe: PrimaryRecipe
    retained_source_days: tuple[str, ...]
    source_weights: SourceWeights | None
    primary: FittedPrimary


def _retained_masses(recipe, original_sources, retained_sources):
    if not retained_sources:
        return ()
    if original_sources == retained_sources:
        return recipe.source_masses
    original = dict(zip(original_sources, recipe.source_masses))
    # fsum makes the total explicit; an unrepresentable total/redistributed
    # positive mass is a numerical failure, never a source-drop fallback.
    total = math.fsum(recipe.source_masses)
    retained = tuple(original[day] for day in retained_sources)
    scale = max(retained)
    proportions = tuple(value / scale for value in retained)
    denominator = math.fsum(proportions)
    masses = tuple(total * (value / denominator) for value in proportions)
    if any(not math.isfinite(value) or value <= 0 for value in masses):
        raise FloatingPointError("omission cannot preserve finite positive source masses")
    return masses


def fit_primary_control(days, *, recipe, control, weight_recipe=None, check):
    """Refit one named intervention at the supplied frozen candidate recipe.

    Source positions are 1-based chronological positions in the original three
    sources. Caller supplies all original packs, even for remove_history; their
    omitted value arrays are never inspected. No partial roster is admitted as
    an ordinary candidate, and 'none' is not an ordinary recipe adjustment.
    """
    _check(check)
    control = validate_control(control)
    _validate_recipe(recipe)  # Require the original, ordinary candidate grammar.
    original = bind_fit_days(days)
    _masses(recipe, original)  # Validate original mass ownership/count before values.
    target = original[-1]
    retained_sources = source_days(target.animal, target.stream, _control=control)
    retained = tuple(day for day in original if day.day in retained_sources or day.day == target.day)
    retained = bind_fit_days(retained, _control=control)
    effective = replace(recipe, adjustment="none") if control.kind == "remove_nuisance" else recipe
    weights = None
    if weight_recipe is not None:
        _neutral_masses(recipe)
        weight_recipe = _weight_recipe(weight_recipe)
        if target.stream != "history_assisted":
            raise PermissionError("today-only controls cannot use a source-weight recipe")
        if retained_sources:
            fitted = _fit_weighted_primary_model(retained, recipe=effective,
                weight_recipe=weight_recipe, check=check, _control=control)
            effective, weights, primary = fitted.effective_recipe, fitted.source_weights, fitted.primary
        else:
            effective = replace(effective, source_masses=(), calibration_mass=weight_recipe.target_mass)
            primary = _fit_primary_model(retained, recipe=effective, check=check, _control=control)
    else:
        effective = replace(effective, source_masses=_retained_masses(recipe,
            tuple(day.day for day in original[:-1]), retained_sources))
        primary = _fit_primary_model(retained, recipe=effective, check=check, _control=control)
    _check(check)
    return FittedPrimaryControl(control, recipe, weight_recipe, effective,
                                retained_sources, weights, primary)


def predict_primary_control_development(fitted, day, *, check):
    """Delegate to the frozen primary; do not refit or introduce new roles."""
    _check(check)
    if not isinstance(fitted, FittedPrimaryControl):
        raise TypeError("one fitted primary control is required")
    return predict_primary_development(fitted.primary, day, check=check)
