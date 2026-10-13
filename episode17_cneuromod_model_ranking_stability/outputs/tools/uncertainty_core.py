"""Fixed EP17 crossed bootstrap and interval decisions; no data reader/refit.

The caller freezes the component family and supplies a callback that recomputes
its declared statistics from participant and whole-concept-block multiplicities.
This module cannot enforce the caller's image identities or private data roles.
"""
from dataclasses import dataclass

import numpy as np

DRAWS = 10_000
SEED = 1729
ORDER_STATISTIC_ONE_BASED = 9501
MARGINS = {"raw_R2": .01, "ceiling_normalized_R2": .05,
           "heldout_correlation": .02, "linear_CKA": .02,
           "beta_high_minus_low_gain": .01}


class UncertaintyError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise UncertaintyError(code)


def fixed(value):
    value = np.array(value, dtype=np.float64, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class SimultaneousBounds:
    component_ids: tuple
    score_scales: tuple
    estimate: np.ndarray
    standard_error: np.ndarray
    critical_value: float
    lower: np.ndarray
    upper: np.ndarray
    margins: np.ndarray
    phase: str
    primary_endpoint_ids: tuple = ()


def crossed_bounds(statistic, component_ids, score_scales, phase, *, primary_endpoint_ids=None):
    """Four participants crossed with24 development or12 audit blocks.

    The callback receives integer multiplicity vectors of lengths4 and24/12.
    It must pool frozen image-level error sums before computing R², retain
    ROI/participant equality, and return exactly the frozen ordered components.
    No invalid draw is skipped, redrawn or assigned a variance floor. When
    explanatory components share the joint family, primary_endpoint_ids must
    identify the complete primary subset before any statistic is evaluated.
    """
    ids, scales = tuple(component_ids), tuple(score_scales)
    require(ids and all(isinstance(x, str) and x for x in ids)
            and len(set(ids)) == len(ids), "component_identity_invalid")
    require(len(ids) == len(scales) and all(x in MARGINS for x in scales), "score_scale_invalid")
    primary = ids if primary_endpoint_ids is None else tuple(primary_endpoint_ids)
    require(primary and len(set(primary)) == len(primary)
            and all(key in ids for key in primary), "primary_endpoint_family_invalid")
    # The beta high/low contrast has raw-R² units, despite its distinct
    # scientific component name and separately declared margin.
    kinds = {"raw_R2" if x == "beta_high_minus_low_gain" else x for x in scales}
    require(len(kinds) == 1 or kinds == {"raw_R2", "ceiling_normalized_R2"},
            "unregistered_mixed_scale_family")
    require(phase in ("development", "audit"), "phase_invalid")
    blocks = 24 if phase == "development" else 12

    def evaluate(participants, concept_blocks):
        participants.setflags(write=False)
        concept_blocks.setflags(write=False)
        raw = np.asarray(statistic(participants, concept_blocks))
        require(raw.dtype.kind in "iuf" and raw.shape == (len(ids),), "statistic_family_changed")
        value = np.asarray(raw, dtype=np.float64)
        require(np.isfinite(value).all(), "undefined_statistic_no_redraw")
        return value.copy()

    estimate = evaluate(np.ones(4, dtype=np.int64), np.ones(blocks, dtype=np.int64))
    rng = np.random.Generator(np.random.PCG64(SEED))
    samples = np.empty((DRAWS, len(ids)), dtype=np.float64)
    for draw in range(DRAWS):
        # Required call order: participant indices first, then block indices.
        subject_weights = np.bincount(rng.integers(0, 4, size=4), minlength=4)
        block_weights = np.bincount(rng.integers(0, blocks, size=blocks), minlength=blocks)
        samples[draw] = evaluate(subject_weights, block_weights)
    require(np.any(samples != samples[0], axis=0).all(), "degenerate_bootstrap_standard_error")
    standard_error = np.std(samples, axis=0, ddof=1)
    require(np.isfinite(standard_error).all() and (standard_error > 0).all(), "degenerate_bootstrap_standard_error")
    max_t = np.max(np.abs((samples - estimate) / standard_error), axis=1)
    require(np.isfinite(max_t).all(), "bootstrap_max_t_nonfinite")
    index = ORDER_STATISTIC_ONE_BASED - 1
    critical = float(np.partition(max_t, index)[index])
    radius = critical * standard_error
    require(np.isfinite(radius).all(), "bootstrap_radius_nonfinite")
    return SimultaneousBounds(ids, scales, fixed(estimate), fixed(standard_error), critical,
                              fixed(estimate - radius), fixed(estimate + radius),
                              fixed([MARGINS[x] for x in scales]), phase, primary)


def relation_decision(bounds, directed_edges):
    """Classify the PRIMARY ENDPOINT subset frozen before joint inference.

    Endpoint adjacency and family membership must be frozen separately.

    Stability and equivalence concern every frozen eligible primary endpoint.
    Joint bounds may also cover explanatory high/low components; those retain
    max-t protection but do not become primary endpoints. This function does
    not establish explanation success or audit acceptance. A
    genuine reversal requires opposite-margin simultaneous bounds at both
    endpoints of one supplied isolated edge, not just opposite point estimates.
    """
    require(isinstance(bounds, SimultaneousBounds), "bounds_required")
    n = len(bounds.component_ids)
    lower, upper, margin = bounds.lower, bounds.upper, bounds.margins
    require(lower.shape == upper.shape == margin.shape == (n,)
            and np.isfinite(lower).all() and np.isfinite(upper).all()
            and (lower <= upper).all() and (margin > 0).all(), "bounds_invalid")
    locations = {key: i for i, key in enumerate(bounds.component_ids)}
    primary = bounds.primary_endpoint_ids or bounds.component_ids
    require(primary and len(set(primary)) == len(primary)
            and all(key in locations for key in primary), "primary_endpoint_family_invalid")
    primary_indices = [locations[key] for key in primary]
    edges = tuple(directed_edges)
    require(all(isinstance(edge, (tuple, list)) and len(edge) == 2
                and edge[0] in primary and edge[1] in primary and edge[0] != edge[1]
                for edge in edges), "isolated_edge_identity_invalid")
    if np.all(lower[primary_indices] > margin[primary_indices]):
        return "stable_positive"
    if np.all(upper[primary_indices] < -margin[primary_indices]):
        return "stable_negative"
    for left, right in edges:
        i, j = locations[left], locations[right]
        if (lower[i] > margin[i] and upper[j] < -margin[j]) or (upper[i] < -margin[i] and lower[j] > margin[j]):
            return "genuine_reversal"
    if np.all(lower[primary_indices] >= -margin[primary_indices]) and np.all(upper[primary_indices] <= margin[primary_indices]):
        return "practical_equivalence"
    return "underidentified"


def audit_precision_adequate(bounds):
    require(isinstance(bounds, SimultaneousBounds) and bounds.phase == "audit", "audit_bounds_required")
    return bool(np.all(bounds.upper - bounds.lower <= 2 * bounds.margins))
