"""EP19 registered logistic-hazard baseline over caller-prepared context features.

This is the source-specific model wrapper, not a sensor feature extractor.
Sensor summaries, cue availability, units, clocks, missing-history/missingness
handling and raw-source admission remain the caller's unfinished responsibilities.
No feature statistic, imputation, sensor mapping, onset label or sampling rate is
chosen here. Fixed feature names/order and declared availability/boundary metadata
prevent interface mix-ups; they cannot prove how upstream features were made.

The wrapper reuses the existing18-head censored logistic fitter, fitting-only
standardization and whole-person/whole-series roles. Its frozen19-category output
is suitable as the non-neural offset for the existing classical EEG backends.
This implements one of three required baseline families, not baseline selection,
forecasting authorization, a final-role opening or an empirical result.
Development inner folds and explicit final baseline-refit scopes are supported;
scope validation does not grant source access or open final evaluation.
"""
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from classical_forecast_backend import (
    FittingScope, FittingStandardizer, GroupIdentity, UnresolvedFit,
    fit_logistic_hazard, hazard_log_probabilities,
)

FAMILY = "regularized_logistic_discrete_hazard"


@dataclass(frozen=True)
class LogisticHazardRecipe:
    recipe_id: str
    source: str
    l2: float
    seeds: tuple[int, ...]
    family: str = FAMILY


def logistic_hazard_recipes(registry, *, source):
    """Enumerate only the three existing source-specific L2 recipes."""
    definition = registry["non_neural_baselines"]
    if source not in ("WAY", "self_paced") or source not in definition["sources"]:
        raise ValueError("one of the two registered baseline sources is required")
    settings = definition["families"][FAMILY]
    if settings != [{"l2": .001}, {"l2": .01}, {"l2": .1}]:
        raise ValueError("the three unchanged logistic-hazard recipes are required")
    seeds = tuple(registry["seeds"])
    if seeds != (1901, 1902, 1903):
        raise ValueError("the registered seed bank must remain unchanged")
    return tuple(LogisticHazardRecipe(f"{source}.{FAMILY}.c{index:02d}", source,
                    setting["l2"], seeds) for index, setting in enumerate(settings))


def _schema(value):
    if isinstance(value, str):
        raise ValueError("feature schema must be a sequence of names, not one string")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("a fixed nonempty feature schema with unique names is required")
    return names


def _canonical_scope(registry, scope, source, *, allow_final_refit=False):
    if not isinstance(scope, FittingScope) or scope.source != source:
        raise ValueError("canonical fitting scope must match the baseline source")
    if source == "WAY" and len(scope.allowed_participants) != 1:
        raise ValueError("one participant per WAY baseline fit is required")
    participant = scope.allowed_participants[0] if source == "WAY" else None
    final_id = (f"WAY.P{participant}.final_refit_series1_7" if source == "WAY"
                else "self_paced.final_refit_all15")
    is_final = allow_final_refit and scope.fold_id == final_id
    if is_final:
        expected = FittingScope.final_refit(registry, source=source, participant=participant)
    else:
        expected = FittingScope.development(registry, source=source, fold_id=scope.fold_id,
                                            participant=participant)
    if scope != expected:
        role = "final refit cohort" if is_final else "development fold"
        raise ValueError(f"fitting scope differs from the canonical {role}")
    return expected


@dataclass(frozen=True)
class ContextFeatures:
    """One prepared non-neural vector and its explicit upstream metadata.

    `available_through` is the latest availability time used by *any* feature,
    including an availability/age/cue feature. The vector belongs to exactly one
    reset boundary. These are caller-supplied facts, not raw-source verification.
    A source's physically absent channels must already be omitted from its
    schema; transient missingness must already follow its admitted feature recipe.
    This backend never fills a NaN, creates a missingness flag or changes a schema.
    """
    values: Any
    feature_schema: tuple[str, ...]
    source: str
    participant: Any
    boundary_id: str
    available_through: float

    def validated(self, *, schema, source, participant, boundary_id, decision_time):
        if self.source != source or _schema(self.feature_schema) != schema:
            raise ValueError("source and feature order must match the fitted baseline schema")
        if self.participant is None or self.participant != participant:
            raise ValueError("context participant must match the example or participant-specific model")
        if not boundary_id or self.boundary_id != boundary_id:
            raise ValueError("context vector crosses the declared reset boundary")
        if (not np.isfinite(decision_time) or not np.isfinite(self.available_through)
                or self.available_through > decision_time):
            raise ValueError("context features must be available at or before the decision")
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(schema),) or not np.all(np.isfinite(values)):
            raise ValueError("finite prepared values for every declared feature are required; no imputation")
        return values


@dataclass(frozen=True)
class BaselineFittingExample:
    context: ContextFeatures
    decision_time: float
    identity: GroupIdentity
    observed_category: int
    complete_bins: int


class LogisticHazardModel:
    """Fitting-fold baseline with18conditional logits and19category probabilities.

    Every fit starts from the existing fitter's deterministic zero initialization.
    The declared seed is retained for matched-run identity; it does not select a
    stochastic optimization path or trigger additional seed searches. Baselines
    are independently fitted per WAY participant or inside each self-paced
    whole-participant training fold, as required by the current registry.
    """
    def fit(self, batches: Iterable[Iterable[BaselineFittingExample]], *,
            recipe: LogisticHazardRecipe, seed: int, registry: Mapping,
            fitting_scope: FittingScope, feature_schema):
        self.fitted = False
        recipes = {item.recipe_id: item for item in logistic_hazard_recipes(registry, source=recipe.source)}
        if recipe.recipe_id not in recipes or recipe != recipes[recipe.recipe_id] or seed not in recipe.seeds:
            raise ValueError("exact registered source-specific recipe and seed required")
        scope = _canonical_scope(registry, fitting_scope, recipe.source, allow_final_refit=True)
        schema = _schema(feature_schema)
        # Scope is admitted before consuming caller batches or fitting statistics.
        examples = tuple(example for batch in batches for example in batch)
        weights = scope.validate([example.identity for example in examples])
        matrix = np.asarray([example.context.validated(schema=schema, source=recipe.source,
            participant=example.identity.participant,
            boundary_id=example.identity.boundary_id, decision_time=example.decision_time)
            for example in examples], dtype=np.float64)
        self.standardizer = FittingStandardizer().fit(matrix, weights)
        features = self.standardizer.transform(matrix)
        self.fit_result = fit_logistic_hazard(features,
            [example.observed_category for example in examples],
            [example.complete_bins for example in examples], weights, penalty=recipe.l2)
        self.recipe, self.seed, self.feature_schema = recipe, seed, schema
        self.fitting_scope = scope
        self.fitted = bool(self.fit_result.converged)
        self.boundary_id = None
        self.last_decision_time = -np.inf
        return {"converged": self.fitted, "iterations": self.fit_result.iterations,
            "objective": self.fit_result.objective, "reason": self.fit_result.reason,
            "recipe_id": recipe.recipe_id, "source": recipe.source, "seed": seed,
            "fold_id": scope.fold_id, "examples": len(examples), "features": len(schema),
            "feature_schema": list(schema), "source_admission_established_by_backend": False}

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("an explicit nonempty reset boundary is required")
        self.boundary_id = boundary_id
        self.last_decision_time = -np.inf

    def predict_log_probabilities(self, context: ContextFeatures, *, decision_time):
        if not getattr(self, "fitted", False):
            raise UnresolvedFit("a nonconverged or incomplete baseline cannot supply EEG offsets")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        # WAY weights belong to one person even after resetting to a new series.
        # Self-paced weights are a group model and must generalize to development
        # validation people outside the fitting fold. Their access role is still
        # admitted by the caller, not inferred from this feature vector.
        participant = (self.fitting_scope.allowed_participants[0]
                       if self.recipe.source == "WAY" else context.participant)
        values = context.validated(schema=self.feature_schema, source=self.recipe.source, participant=participant,
            boundary_id=self.boundary_id, decision_time=decision_time)
        features = self.standardizer.transform(values[None, :])
        logits = features @ self.fit_result.coefficients + self.fit_result.intercept
        log_probabilities = hazard_log_probabilities(logits)[0]
        self.last_decision_time = decision_time
        return log_probabilities
