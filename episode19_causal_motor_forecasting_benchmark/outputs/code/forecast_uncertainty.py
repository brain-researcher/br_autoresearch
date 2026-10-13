"""Registered EP19 factorial-equivalence intervals on supplied participant means.

The caller supplies already matched natural-log scores: identical eligible rows,
selected recipes, ensemble rules and adaptation conditions within every required
comparison. Means pool each person's eligible rows before equal-person inference.
This library cannot establish that upstream matching, source admission, or a
prospective freeze of the supplied seed and margins. It performs no file I/O,
model fitting, source qualification or held-out evaluation admission.

All forty endpoints share one unstudentized maximum-absolute-centered bootstrap
statistic. Equivalence here concerns only interval containment; mutation checks,
decisive contrasts and replication remain separate explanation requirements.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from forecast_scoring import ScoringScope, UnresolvedScore


SOURCES = ("WAY", "self_paced")
FAMILIES = (
    "past_only_filterbank_bandpower_elastic_net",
    "past_only_covariance_riemannian_tangent_model",
    "past_only_compact_EEGNet",
    "dilated_past_only_temporal_convolutional_network",
    "small_one_directional_GRU",
)
HORIZONS = ("near_event", "primary_strict")
CONDITIONS = ("original_full_reference", "Q11", "Q01", "Q10", "Q00")
ENDPOINTS = ("Q11_minus_original_full_reference", "factorial_interaction")
PARTICIPANTS = {
    "WAY": tuple(range(1, 13)),
    "self_paced": ("sub-01", "sub-05", "sub-07", "sub-10", "sub-13", "sub-16", "sub-19", "sub-23"),
}
DRAWS = 10_000
CONFIDENCE = .95
DRAW_BATCH = 256


@dataclass(frozen=True)
class FactorialScorePanel:
    """Explicit axes for participant × family × horizon × condition scores.

Every axis uses the canonical order exported by this module, including the
registered participant order. No implicit alignment, replacement or row dropping
occurs. The array is caller-owned and is never modified.
"""
    source: str
    participants: tuple
    families: tuple
    horizons: tuple
    conditions: tuple
    scores: np.ndarray


def _margin(value):
    if value is None:
        return None
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError("an externally frozen positive finite margin or None is required")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("margin must be finite in float64")
    return value


def _panels(registry, panels):
    if tuple(registry["reference_grid"]["family_variants"]) != FAMILIES:
        raise ValueError("the five canonical registered reference families are required")
    if registry["matched_conditions"]["factorial"] != {
            "Q11": [1, 1], "Q01": [0, 1], "Q10": [1, 0], "Q00": [0, 0]}:
        raise ValueError("the registered four factorial conditions are required")
    panels = tuple(panels)
    if (len(panels) != 2 or any(not isinstance(panel, FactorialScorePanel) for panel in panels)
            or any(not isinstance(panel.source, str) for panel in panels)
            or {panel.source for panel in panels} != set(SOURCES)):
        raise UnresolvedScore("exactly one complete panel for each registered source is required")
    lookup = {panel.source: panel for panel in panels}
    values = {}
    for source in SOURCES:
        scope = ScoringScope.from_registry(registry, source=source, stage="held_out")
        if scope.participants != PARTICIPANTS[source]:
            raise ValueError("registry must retain the exact registered held-out participant IDs")
        panel = lookup[source]
        people = tuple(panel.participants)
        if len(people) != len(scope.participants):
            raise UnresolvedScore("complete registered participant cohort required; no successful subset")
        correct_types = (all(isinstance(person, Integral) and not isinstance(person, (bool, np.bool_))
                             for person in people) if source == "WAY"
                         else all(isinstance(person, str) for person in people))
        if not correct_types or people != scope.participants:
            raise ValueError("exact registered participant IDs and canonical order are required")
        if (tuple(panel.families) != FAMILIES or tuple(panel.horizons) != HORIZONS
                or tuple(panel.conditions) != CONDITIONS):
            raise ValueError("complete canonical family, horizon and condition axes are required")
        raw = np.asarray(panel.scores)
        if raw.shape != (len(people), 5, 2, 5) or raw.dtype.kind not in "fiu":
            raise ValueError("real numeric participant-by-family-by-horizon-by-condition scores required")
        scores = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError("all registered participant scores must be finite")
        # Both endpoints are paired within participant BEFORE means or resampling.
        with np.errstate(over="raise", invalid="raise"):
            try:
                endpoints = np.stack((scores[..., 1] - scores[..., 0],
                    scores[..., 1] - scores[..., 3] - scores[..., 2] + scores[..., 4]), axis=-1)
            except FloatingPointError as exc:
                raise ValueError("factorial endpoint arithmetic is not finite") from exc
        values[source] = endpoints.reshape(len(people), 20)
    return values


def factorial_equivalence_intervals(*, registry, panels, seed,
                                   full_reference_margin, interaction_margin):
    """Return the single registered 40-endpoint simultaneous interval family.

``seed`` is a required nonnegative integer frozen externally before evaluation.
Two PCG64 streams, spawned from that seed in WAY/self_paced order, independently
resample complete source cohorts. Each draw's indices are shared across that
source's twenty endpoints. The 95% quantile uses ``method='higher'``. Batch size
only bounds working memory; no source or endpoint is omitted from the maximum.

Missing margins yield unresolved calls while retaining intervals. Otherwise
every interval must be strictly inside (-margin, +margin); equality does not
establish equivalence. Failure to establish equivalence is not evidence of
non-equivalence. No broader temporal-explanation decision is made.
"""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("an explicit prospectively frozen nonnegative integer seed is required")
    seed = int(seed)
    margins = (_margin(full_reference_margin), _margin(interaction_margin))
    values = _panels(registry, panels)
    means = {}
    maximum = np.zeros(DRAWS, dtype=np.float64)
    children = np.random.SeedSequence(seed).spawn(len(SOURCES))
    try:
        with np.errstate(over="raise", invalid="raise"):
            for source, child in zip(SOURCES, children):
                table = values[source]
                means[source] = table.mean(axis=0)
                generator = np.random.Generator(np.random.PCG64(child))
                for first in range(0, DRAWS, DRAW_BATCH):
                    last = min(first + DRAW_BATCH, DRAWS)
                    indices = generator.integers(0, len(table), size=(last - first, len(table)))
                    centered = table[indices].mean(axis=1) - means[source]
                    maximum[first:last] = np.maximum(maximum[first:last], np.abs(centered).max(axis=1))
            radius = float(np.quantile(maximum, CONFIDENCE, method="higher"))
            bounds = {source: (means[source] - radius, means[source] + radius) for source in SOURCES}
    except FloatingPointError as exc:
        raise ValueError("bootstrap interval arithmetic is not finite") from exc

    intervals = []
    for source in SOURCES:
        lower, upper = bounds[source]
        for family_index, family in enumerate(FAMILIES):
            for horizon_index, horizon in enumerate(HORIZONS):
                for endpoint_index, endpoint in enumerate(ENDPOINTS):
                    index = (family_index * len(HORIZONS) + horizon_index) * len(ENDPOINTS) + endpoint_index
                    margin = margins[endpoint_index]
                    intervals.append({
                        "source": source, "family": family, "horizon": horizon, "endpoint": endpoint,
                        "participant_count": len(PARTICIPANTS[source]),
                        "mean": float(means[source][index]),
                        "lower": float(lower[index]), "upper": float(upper[index]),
                        "margin": margin,
                        "inside_margin": (None if margin is None else
                                          bool(lower[index] > -margin and upper[index] < margin)),
                    })
    calls = {}
    for endpoint, margin in zip(ENDPOINTS, margins):
        selected = [item for item in intervals if item["endpoint"] == endpoint]
        established = None if margin is None else all(item["inside_margin"] for item in selected)
        calls[endpoint] = {
            "status": ("unresolved_missing_margin" if established is None else
                       "established" if established else "not_established"),
            "equivalence_established": established,
            "margin": margin, "endpoint_count": len(selected),
        }
    return {
        "sources": SOURCES, "participants": PARTICIPANTS.copy(), "families": FAMILIES,
        "horizons": HORIZONS, "endpoint_count": len(intervals), "draws": DRAWS,
        "confidence_level": CONFIDENCE, "seed": seed,
        "random_generator": "PCG64 with SeedSequence-spawned independent WAY/self_paced streams",
        "quantile_method": "higher", "simultaneous_half_width": radius,
        "statistic": "maximum_absolute_centered_equal_participant_mean_over_all_40_endpoints",
        "score_units": "natural_log_probability_per_eligible_decision_higher_is_better",
        "intervals": intervals,
        "full_reference_equivalence": calls[ENDPOINTS[0]],
        "interaction_equivalence": calls[ENDPOINTS[1]],
        "full_temporal_explanation_evaluated": False,
        "interpretation_limit": "Interval containment only; matching, source admission, mutation checks, decisive contrasts and replication remain upstream or separate.",
    }
