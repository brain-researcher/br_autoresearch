"""Registered EP19 horizon scores for an already admitted matched prediction set.

No source reader, risk/onset construction, seed selection, calibration, bootstrap
or scientific decision occurs here. The caller supplies aligned post-ensemble
predictions for one model recipe and adaptation condition. Source admission,
completed folds and the full natural-prevalence row roster remain upstream.
An explicit scope describes the comparison; it does not authorize held-out I/O.
"""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from classical_forecast_backend import _labels, _matrix, participant_weights
from ensemble_forecast_backend import _row_keys
from model_recipes import self_paced_folds


HORIZONS = {"near_event": (0, 6), "primary_strict": (6, 12),
            "longer_diagnostic": (12, 18)}


class UnresolvedScore(RuntimeError):
    """A required participant or matched condition has no complete score."""


@dataclass(frozen=True)
class ScoringScope:
    source: str
    stage: str
    participants: tuple
    allowed_series: tuple | None
    surrogates: tuple

    @classmethod
    def from_registry(cls, registry, *, source, stage):
        folds = registry["folds"]
        if source == "WAY" and stage in ("development", "feedback", "held_out"):
            participants = tuple(folds["WAY_participants"])
            series_key = {"development": "WAY_inner_validation_series",
                          "feedback": "WAY_feedback_series", "held_out": "WAY_held_out_series"}[stage]
            series = tuple(folds[series_key])
            if series != {"development": (1, 2, 3, 4, 5, 6), "feedback": (7,), "held_out": (8, 9)}[stage]:
                raise ValueError("registered WAY scoring series changed")
            required_count = 12
        elif source == "self_paced" and stage in ("development", "held_out"):
            development = tuple(person for fold in self_paced_folds(registry) for person in fold.validation)
            participants = (tuple(sorted(development)) if stage == "development"
                            else tuple(folds["self_paced_held_out"]))
            series, required_count = None, 15 if stage == "development" else 8
        else:
            raise ValueError("explicit registered source and scoring stage required")
        if len(participants) != required_count or len(set(participants)) != required_count:
            raise ValueError("complete registered scoring cohort required")
        matched = registry["matched_conditions"]
        surrogates = (tuple(name for name in matched["primary"] if name not in ("real_EEG", "zero_EEG"))
                      if stage == "held_out" else tuple(matched["development_surrogates"]))
        expected = (tuple(f"shift_{i}" for i in range(1, 5)) + tuple(f"phase_{i}" for i in range(1, 5))
                    if stage == "held_out" else ("shift_1", "phase_1"))
        if surrogates != expected:
            raise ValueError("registered complete surrogate panel changed")
        return cls(source, stage, participants, series, surrogates)

    @property
    def conditions(self):
        return ("non_neural", "real_EEG", "zero_EEG", *self.surrogates)


def _log_probabilities(values, rows):
    if np.iscomplexobj(values):
        raise ValueError("real log probabilities required")
    values = _matrix(values, 19)
    if len(values) != rows or not np.allclose(logsumexp(values, axis=1), 0., rtol=0, atol=1e-12):
        raise ValueError("complete normalized 19-category log probabilities required")
    return values


def horizon_log_scores(log_probabilities, observed_category, observed_through_bins, *, horizon):
    """Return score, eligibility and binary outcome on one registered horizon.

Scores are natural logarithms, higher is better. Coverage is a separate label-
builder output: the number of complete 50-ms follow-up bins, capped at 18,
through which onset status is observed. It is required even for observed events;
the training-only ``complete_bins`` convention cannot supply it implicitly.
An observed early event does not waive the full-upper-edge observation rule.
Ineligible scores and outcomes are NaN, never negative labels or zero scores.
"""
    if horizon not in HORIZONS:
        raise ValueError("one registered horizon must be named explicitly")
    categories, coverage = _labels(observed_category, observed_through_bins, len(observed_category))
    logp = _log_probabilities(log_probabilities, len(categories))
    # An onset in interval j cannot be observed before the j completed bins
    # preceding it. Category 18 certifies the entire 900-ms non-onset period.
    if np.any((categories >= 0) & (coverage < categories)):
        raise ValueError("onset category is inconsistent with observed follow-up coverage")
    lower, upper = HORIZONS[horizon]
    eligible = coverage >= upper
    positive = (categories >= lower) & (categories < upper)
    scores = np.full(len(categories), np.nan)
    outcomes = np.full(len(categories), np.nan)
    selected = logp[eligible]
    # Summing the explicit complement avoids log(1-p) cancellation and retains
    # very small finite probabilities without clipping or an epsilon floor.
    log_positive = logsumexp(selected[:, lower:upper], axis=1)
    outside = np.r_[np.arange(lower), np.arange(upper, 19)]
    log_negative = logsumexp(selected[:, outside], axis=1)
    scores[eligible] = np.where(positive[eligible], log_positive, log_negative)
    outcomes[eligible] = positive[eligible].astype(float)
    return {"log_scores": scores, "eligible": eligible, "binary_outcomes": outcomes}


def score_matched_forecasts(*, registry, scope, horizon, expected_row_keys,
                           condition_log_probabilities, condition_row_keys,
                           observed_category, observed_through_bins):
    """Score one complete source/stage/recipe/adaptation/horizon comparison.

All conditions use the same caller-admitted rows, labels and follow-up mask.
Pool every eligible series/run row within a participant, then give each person
equal weight. Choose the strongest complete surrogate separately for each
participant at this horizon; a near-event winner never selects the strict
comparator. Return continuous scores/increments only, with no margin or call.
"""
    if not isinstance(scope, ScoringScope) or scope != ScoringScope.from_registry(
            registry, source=scope.source, stage=scope.stage):
        raise ValueError("canonical registered scoring scope required")
    rows = _row_keys(expected_row_keys)
    bindings = {}
    seen = set()
    for identity, _ in rows:
        if (identity.source != scope.source or identity.participant not in scope.participants
                or (scope.allowed_series is not None and identity.series_or_run not in scope.allowed_series)):
            raise ValueError("prediction row crosses the declared scoring scope")
        binding = (identity.source, identity.participant, identity.series_or_run)
        if identity.boundary_id in bindings and bindings[identity.boundary_id] != binding:
            raise ValueError("one reset boundary names different recording groups")
        bindings[identity.boundary_id] = binding
        seen.add(identity.participant)
    if seen != set(scope.participants):
        raise UnresolvedScore("complete registered participant cohort required; no successful subset")
    names = scope.conditions
    for supplied in (condition_log_probabilities, condition_row_keys):
        if not isinstance(supplied, Mapping) or set(supplied) != set(names):
            raise UnresolvedScore("every matched condition and registered surrogate is required")
    for name in names:
        if _row_keys(condition_row_keys[name]) != rows:
            raise ValueError("every matched condition must use the exact admitted row order")
    categories, coverage = _labels(observed_category, observed_through_bins, len(rows))
    row_results = {name: horizon_log_scores(condition_log_probabilities[name], categories, coverage,
                                          horizon=horizon) for name in names}
    eligible = row_results[names[0]]["eligible"]
    people = np.asarray([identity.participant for identity, _ in rows])
    participant_counts = {person: int(np.count_nonzero(eligible & (people == person)))
                          for person in scope.participants}
    if any(count == 0 for count in participant_counts.values()):
        raise UnresolvedScore("every registered participant needs observed support at this horizon")
    participant_scores = {
        name: np.asarray([row_results[name]["log_scores"][eligible & (people == person)].mean()
                          for person in scope.participants]) for name in names}
    # Reuse the existing equal-person/natural-prevalence weighting primitive.
    weights = participant_weights(people[eligible])
    macro = {name: float(weights @ row_results[name]["log_scores"][eligible]) for name in names}
    surrogate_matrix = np.stack([participant_scores[name] for name in scope.surrogates])
    strongest = surrogate_matrix.max(axis=0)
    winners = tuple(tuple(name for index, name in enumerate(scope.surrogates)
                          if surrogate_matrix[index, person] == strongest[person])
                    for person in range(len(scope.participants)))
    real = participant_scores["real_EEG"]
    increments = {"real_minus_non_neural": real - participant_scores["non_neural"],
                  "real_minus_zero_EEG": real - participant_scores["zero_EEG"],
                  "real_minus_strongest_surrogate": real - strongest}
    return {"source": scope.source, "stage": scope.stage, "horizon": horizon,
            "horizon_ms": tuple(edge * 50 for edge in HORIZONS[horizon]),
            "participants": scope.participants, "conditions": names,
            "eligible_row_count": int(eligible.sum()), "excluded_row_count": int((~eligible).sum()),
            "participant_row_counts": participant_counts, "eligible": eligible.copy(),
            "binary_outcomes": row_results[names[0]]["binary_outcomes"].copy(),
            "participant_log_scores": participant_scores, "macro_log_scores": macro,
            "strongest_surrogate_log_scores": strongest, "strongest_surrogate_names": winners,
            "participant_increments": increments,
            "macro_increments": {name: float(values.mean()) for name, values in increments.items()},
            "score_units": "natural_log_probability_per_eligible_decision_higher_is_better"}
