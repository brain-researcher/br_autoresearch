"""Finite EP19 recipes, grouped folds and the future model interface.

No source I/O, numerical framework import, fit, detector or forecast occurs
here. The registry is a prospective implementation specification, not source
qualification. Model backends must use forecast_components' existing censored
likelihood/calibration and receive admitted timing/filter semantics explicitly.
"""
from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class ReferenceRecipe:
    recipe_id: str
    family: str
    history_seconds: float
    regularization: float
    family_parameters: Mapping[str, Any]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class GroupedFold:
    fold_id: str
    fit: tuple[Any, ...]
    validation: tuple[Any, ...]
    grouping: str


def reference_recipes(registry):
    grid = registry["reference_grid"]
    seeds = tuple(registry["seeds"])
    for family, variants in grid["family_variants"].items():
        for index, (history, penalty, variant) in enumerate(product(
                grid["history_seconds"], grid["regularization"], variants)):
            yield ReferenceRecipe(f"{family}.c{index:02d}", family,
                                  history, penalty, dict(variant), seeds)


def self_paced_folds(registry):
    groups = registry["folds"]["self_paced_validation_participants"]
    everyone = tuple(person for group in groups for person in group)
    if len(groups) != 5 or len(everyone) != 15 or len(set(everyone)) != 15:
        raise ValueError("Require exactly five disjoint three-person development folds")
    if any(len(group) != 3 for group in groups):
        raise ValueError("Every development validation fold contains three people")
    if set(everyone) & set(registry["folds"]["self_paced_held_out"]):
        raise ValueError("Held-out participants cannot enter development folds")
    return tuple(GroupedFold(f"self_paced.f{index}",
        tuple(sorted(set(everyone) - set(group))), tuple(group),
        "whole_participant_all_sessions_runs_and_state")
        for index, group in enumerate(groups))


def way_inner_folds(registry, participant):
    definition = registry["folds"]
    if participant not in definition["WAY_participants"]:
        raise ValueError("Unknown WAY participant; no replacement participant")
    series = tuple(definition["WAY_inner_validation_series"])
    if series != (1, 2, 3, 4, 5, 6):
        raise ValueError("WAY inner selection is restricted to whole series1–6")
    return tuple(GroupedFold(f"WAY.P{participant}.validation_series{held}",
        tuple(s for s in series if s != held), (held,),
        "whole_series_within_one_participant_reset_state_at_each_boundary")
        for held in series)


def reference_jobs(registry, *, final_surrogates=False, factorial=False,
                   selected_recipe_ids=None):
    """Enumerate model identities only; never dispatch or remove failed cells."""
    if final_surrogates and factorial:
        raise ValueError("Primary surrogate and factorial panels are distinct")
    recipes = tuple(reference_recipes(registry))
    if final_surrogates or factorial:
        if selected_recipe_ids is None:
            raise ValueError("Frozen one-per-family selections required; no final re-search")
        chosen = tuple(selected_recipe_ids)
        lookup = {recipe.recipe_id: recipe for recipe in recipes}
        if len(chosen) != 5 or len(set(chosen)) != 5 or any(key not in lookup for key in chosen):
            raise ValueError("Exactly five distinct registered selected recipes required")
        recipes = tuple(lookup[key] for key in chosen)
        if len({recipe.family for recipe in recipes}) != 5:
            raise ValueError("Require one selected recipe from every reference family")
    elif selected_recipe_ids is not None:
        raise ValueError("Initial reference search retains all twelve configurations per family")
    if factorial:
        conditions = tuple(registry["matched_conditions"]["factorial"])
    elif final_surrogates:
        conditions = tuple(registry["matched_conditions"]["primary"])
    else:
        conditions = ("real_EEG", "zero_EEG",
                      *registry["matched_conditions"]["development_surrogates"])
    for recipe in recipes:
        for seed, condition in product(recipe.seeds, conditions):
            yield recipe, seed, condition


class ForecastModel(Protocol):
    """Interface for each future family backend, not a dummy fitted model.

    Fit batches include natural at-risk grid weights, observed categories and
    complete censoring bins. They contain one declared fitting role only.
    Semantics explicitly supplies rates, admitted clocks/units, causal filter
    coefficients, warm-up, montage and gap/reset behavior. Unknown entries do
    not acquire defaults here. No detector or provider annotation is inferred.
    """
    def fit(self, batches: Iterable[Any], *, recipe: ReferenceRecipe,
            seed: int, semantics: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def reset(self, *, boundary_id: str) -> None: ...

    def predict_log_probabilities(self, past_input: Any, *, decision_time: float,
            context_log_probabilities: Any) -> Any:
        """Return normalized19-category log probabilities from available past only."""
        ...
