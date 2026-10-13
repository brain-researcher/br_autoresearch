"""Registered EP19 boosted non-neural hazard baseline on prepared features.

No source I/O, timing/unit assumptions, raw feature extraction or access grant.
The existing 18-head/depth2,3,4/200-tree/0.05 recipe is implemented with full-
sample binomial gradient boosting. Explicit engineering settings are zero
initial logits, all features, no class weighting/subsampling/early stopping,
and sklearn's exact greedy Friedman-MSE trees with minimum leaf size one.
The seed controls split tie ordering, not an additional model search.

Only observed conditional risk intervals enter each head. A head with no
observed data or just one class makes the whole fit unresolved before fitting;
there is no invented probability floor, pseudo-event or silent head omission.
Final probabilities use the existing stable hazard-to-19-category transform.
This accepts canonical development folds and explicit final baseline-refit
scopes. Source admission, selection and final evaluation remain separate.
"""
from dataclasses import dataclass

import numpy as np

from classical_forecast_backend import (
    FittingStandardizer, UnresolvedFit, _labels, _matrix, _weights,
    hazard_log_probabilities, hazard_loss_and_gradient,
)
from non_neural_forecast_backend import _canonical_scope, _schema

FAMILY = "gradient_boosted_discrete_hazard"
SEEDS = (1901, 1902, 1903)


@dataclass(frozen=True)
class BoostedHazardRecipe:
    recipe_id: str
    source: str
    depth: int
    trees: int
    learning_rate: float
    seeds: tuple[int, ...]
    family: str = FAMILY


def boosted_hazard_recipes(registry, *, source):
    definition = registry["non_neural_baselines"]
    expected = [{"depth": depth, "trees": 200, "learning_rate": .05} for depth in (2, 3, 4)]
    if (source not in ("WAY", "self_paced") or source not in definition["sources"]
            or definition["families"][FAMILY] != expected or tuple(registry["seeds"]) != SEEDS):
        raise ValueError("unchanged source-specific boosted-hazard recipes and seeds required")
    return tuple(BoostedHazardRecipe(f"{source}.{FAMILY}.c{index:02d}", source,
                 setting["depth"], setting["trees"], setting["learning_rate"], SEEDS)
                 for index, setting in enumerate(expected))


def conditional_hazard_targets(category, complete, rows):
    """The exact conditional Bernoulli observations in the existing censored NLL."""
    category, complete = _labels(category, complete, rows)
    bins = np.arange(18)[None, :]
    observed = np.where(category < 0, complete, np.minimum(category + 1, 18))
    at_risk = bins < observed[:, None]
    events = (bins == category[:, None]) & (category[:, None] >= 0)
    return at_risk, events


def _new_head(depth, seed):
    # Import only when an admitted fitting call actually needs this backend.
    # Zero initialization avoids adopting a clipped class-prior probability.
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(loss="log_loss", learning_rate=.05, n_estimators=200,
        subsample=1., criterion="friedman_mse", min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0., max_depth=depth, min_impurity_decrease=0., init="zero",
        random_state=seed, max_features=None, max_leaf_nodes=None, warm_start=False,
        n_iter_no_change=None, ccp_alpha=0.)


@dataclass(frozen=True)
class BoostedHazardFit:
    heads: tuple
    feature_count: int
    completed: bool
    objective: float | None
    initial_objective: float
    reason: str
    unresolved_head: int | None = None

    def logits(self, features):
        if not self.completed or len(self.heads) != 18:
            raise UnresolvedFit(self.reason)
        features = _matrix(features, self.feature_count)
        result = np.column_stack([head.decision_function(features) for head in self.heads])
        if result.shape != (len(features), 18) or not np.all(np.isfinite(result)):
            raise UnresolvedFit("nonfinite_boosted_hazard_logits")
        return result


def fit_boosted_hazard(features, category, complete, weights, *, depth, seed):
    features = _matrix(features)
    category, complete = _labels(category, complete, len(features))
    weights = _weights(weights, len(features))
    if type(depth) is not int or depth not in (2, 3, 4) or type(seed) is not int or seed not in SEEDS:
        raise ValueError("registered boosted depth and seed required")
    risk, events = conditional_hazard_targets(category, complete, len(features))
    initial = float(weights @ hazard_loss_and_gradient(np.zeros((len(features), 18)), category, complete)[0])
    for index in range(18):
        observed = events[risk[:, index], index]
        reason = ("no_observed_conditional_intervals" if not len(observed) else
                  "single_class_conditional_head_requires_unresolved_fit" if np.unique(observed).size != 2 else None)
        if reason is not None:
            return BoostedHazardFit((), features.shape[1], False, None, initial, reason, index)
    heads = []
    for index in range(18):
        mask = risk[:, index]
        head = _new_head(depth, seed)
        head.fit(features[mask], events[mask, index].astype(np.int64), sample_weight=weights[mask])
        if (head.n_estimators_ != 200 or head.estimators_.shape != (200, 1)
                or not np.array_equal(head.classes_, [0, 1])):
            raise UnresolvedFit("boosted_head_did_not_complete_exact_registered_tree_schedule")
        heads.append(head)
    fitted = BoostedHazardFit(tuple(heads), features.shape[1], True, None, initial, "fixed_200_tree_schedule_complete")
    logits = fitted.logits(features)
    objective = float(weights @ hazard_loss_and_gradient(logits, category, complete)[0])
    if not np.isfinite(objective):
        return BoostedHazardFit((), features.shape[1], False, None, initial, "nonfinite_boosted_objective")
    return BoostedHazardFit(tuple(heads), features.shape[1], True, objective, initial, fitted.reason)


class BoostedHazardModel:
    """The same prepared-context, source/fold and reset interface as logistic."""
    def fit(self, batches, *, recipe, seed, registry, fitting_scope, feature_schema):
        self.fitted = False
        recipes = {item.recipe_id: item for item in boosted_hazard_recipes(registry, source=recipe.source)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or seed not in recipe.seeds:
            raise ValueError("exact registered boosted recipe and seed required")
        scope = _canonical_scope(registry, fitting_scope, recipe.source, allow_final_refit=True)
        schema = _schema(feature_schema)
        examples = tuple(example for batch in batches for example in batch)
        weights = scope.validate([example.identity for example in examples])
        matrix = np.asarray([example.context.validated(schema=schema, source=recipe.source,
            participant=example.identity.participant, boundary_id=example.identity.boundary_id,
            decision_time=example.decision_time) for example in examples], dtype=np.float64)
        self.standardizer = FittingStandardizer().fit(matrix, weights)
        self.fit_result = fit_boosted_hazard(self.standardizer.transform(matrix),
            [example.observed_category for example in examples], [example.complete_bins for example in examples],
            weights, depth=recipe.depth, seed=seed)
        self.recipe, self.seed, self.feature_schema = recipe, seed, schema
        self.fitting_scope = scope
        self.fitted = self.fit_result.completed
        self.boundary_id = None
        self.last_decision_time = -np.inf
        return {"completed": self.fitted, "reason": self.fit_result.reason,
            "unresolved_head": self.fit_result.unresolved_head, "objective": self.fit_result.objective,
            "recipe_id": recipe.recipe_id, "source": recipe.source, "seed": seed,
            "fold_id": scope.fold_id, "examples": len(examples), "features": len(schema),
            "feature_schema": list(schema), "heads": 18, "trees_per_head": 200,
            "source_admission_established_by_backend": False}

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("an explicit nonempty reset boundary is required")
        self.boundary_id = boundary_id
        self.last_decision_time = -np.inf

    def predict_log_probabilities(self, context, *, decision_time):
        if not getattr(self, "fitted", False):
            raise UnresolvedFit("an unresolved or incomplete baseline cannot supply EEG offsets")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        participant = (self.fitting_scope.allowed_participants[0]
                       if self.recipe.source == "WAY" else context.participant)
        values = context.validated(schema=self.feature_schema, source=self.recipe.source, participant=participant,
            boundary_id=self.boundary_id, decision_time=decision_time)
        features = self.standardizer.transform(values[None, :])
        logp = hazard_log_probabilities(self.fit_result.logits(features))[0]
        self.last_decision_time = decision_time
        return logp
