"""Registered EP18 participant-t evidence, without data access or release authority.

Inputs are already normalized, six-outer-fold-averaged participant contrasts.
This module never normalizes losses, selects a model, drops a participant,
projects power, simulates recovery, or opens audit. The caller binds the frozen
roster, baseline and development-calibrated margins before supplying arrays.

The four negative endpoints share alpha=.05 across BOTH tails (eight bounds).
Primary one-sided positivity and two-sided classification are separate
registered families. A constant participant vector has sample SD and SE zero;
its interval is the point mean, without epsilon or a substitute t statistic.

Classification matches below are only the policy's numerical inequalities.
They are NOT episode outcomes: integrity, capacity, reliability, precision,
recovery, influence, access and exposure evidence are intentionally absent.
In particular an adequacy/equivalence inequality cannot assert that the
separately required precision qualification has passed. No gate flags are
accepted from a caller and no scientific/audit authorization is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.stats import t


ALPHA = .05
ROLE_SIZES = {"development": 30, "audit": 16}
PRIMARY = ("categorical_basis_advantage",) + tuple(
    f"partition_specificity_{index}" for index in range(1, 9))
POSITIVE = ("heldout_image_prediction_gain", "target_and_response_prediction_gain",
            "repeated_image_crossvalidated_template_prediction_gain")
NEGATIVE = ("negative_lag_categorical_basis_advantage", "concept_labels_shifted_37_events",
            "concept_labels_shifted_73_events", "concept_labels_shifted_109_events")
LINEAR = ("linear_label_basis_advantage",)


@dataclass(frozen=True)
class Margins:
    """Original formulas applied to previously frozen core-reference G and T.

    Neither gain is estimated here. G<=0 remains evidence of the contract's
    underidentified reference, not a reason to exclude data or raise margins.
    Independent or below-floor thresholds cannot be supplied to this object.
    """
    G: float
    T: float
    delta_C: float = field(init=False)
    delta_P: float = field(init=False)
    delta_Q: float = field(init=False)
    delta_T: float = field(init=False)

    def __post_init__(self):
        for name in ("G", "T"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError("finite_frozen_core_calibration_required")
            object.__setattr__(self, name, float(value))
        for name, value in (("delta_C", max(.001, .05*self.G)),
                            ("delta_P", max(.0005, .025*self.G)),
                            ("delta_Q", max(.002, .25*self.G)),
                            ("delta_T", max(.001, .25*self.T))):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FamilyBounds:
    endpoints: tuple[str, ...]
    participant_count: int
    degrees_of_freedom: int
    familywise_alpha: float
    tail_alpha: float
    critical_t: float
    mean: tuple[float, ...]
    sample_sd: tuple[float, ...]
    standard_error: tuple[float, ...]
    halfwidth: tuple[float, ...]
    lower: tuple[float, ...]
    # None means a registered lower-bound-only family, not +infinity in JSON.
    upper: tuple[float, ...] | None


@dataclass(frozen=True)
class NumericalClassification:
    primary_mean_minus_margin: tuple[float, ...]
    primary_lower_minus_margin: tuple[float, ...]
    primary_upper_minus_margin: tuple[float, ...]
    positive_lower_minus_threshold: tuple[float, ...]
    negative_lower_minus_margin: tuple[float, ...]
    negative_upper_minus_margin: tuple[float, ...]
    partition_equivalent_indices: tuple[int, ...]
    partition_better_than_true_indices: tuple[int, ...]
    negative_control_relation: str
    development_mean_relation: str | None
    # Policy order among the effect inequalities only; never a final decision.
    audit_effect_rule_matches: tuple[str, ...]
    diagnostic_rule_matches: tuple[str, ...]


@dataclass(frozen=True)
class ParticipantEvidence:
    role: str
    participant_ids: tuple[str, ...]
    margins: Margins
    primary_positive: FamilyBounds
    primary_classification: FamilyBounds
    positive_controls: FamilyBounds
    negative_controls: FamilyBounds
    linear_label: FamilyBounds
    numerical: NumericalClassification


def _vectors(values, names, n):
    if not isinstance(values, Mapping) or set(values) != set(names):
        raise ValueError("complete_exact_endpoint_family_required")
    columns = []
    for name in names:
        raw = np.asarray(values[name])
        if raw.shape != (n,) or raw.dtype.kind not in "iuf":
            raise ValueError("one_real_contrast_per_assigned_participant_required")
        data = raw.astype(np.float64, copy=True)
        if not np.isfinite(data).all():
            raise ValueError("all_assigned_participant_contrasts_must_be_finite")
        columns.append(data)
    return np.column_stack(columns)


def _bounds(matrix, names, *, two_sided):
    n, count = matrix.shape
    tail_alpha = ALPHA / (count * (2 if two_sided else 1))
    critical = float(t.ppf(1. - tail_alpha, n - 1))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        constant = np.all(matrix == matrix[0], axis=0)
        means = np.empty(count)
        deviations = np.empty(count)
        means[constant] = matrix[0, constant]
        deviations[constant] = 0.
        if np.any(~constant):
            means[~constant] = np.mean(matrix[:, ~constant], axis=0)
            deviations[~constant] = np.std(matrix[:, ~constant], axis=0, ddof=1)
        se = deviations / np.sqrt(float(n))
        halfwidth = critical * se
        lower = means - halfwidth
        upper = means + halfwidth if two_sided else None
    if not all(np.isfinite(value).all() for value in
               (critical, means, deviations, se, halfwidth, lower, () if upper is None else upper)):
        raise ValueError("participant_interval_not_numerically_representable")
    as_tuple = lambda values: tuple(map(float, values))
    return FamilyBounds(tuple(names), n, n-1, ALPHA, tail_alpha, critical,
        as_tuple(means), as_tuple(deviations), as_tuple(se), as_tuple(halfwidth),
        as_tuple(lower), None if upper is None else as_tuple(upper))


def _numerical(role, margins, primary, classification, positive, negative, linear, development):
    thresholds = np.asarray((margins.delta_C,) + (margins.delta_P,) * 8)
    mean = np.asarray(primary.mean)
    lower, upper = np.asarray(primary.lower), np.asarray(classification.upper)
    classification_lower = np.asarray(classification.lower)
    neg_lower, neg_upper = np.asarray(negative.lower), np.asarray(negative.upper)
    pos_lower = np.asarray(positive.lower)
    pos_thresholds = np.asarray((margins.delta_Q, margins.delta_T, 0.))
    equivalent = tuple(index for index in range(1, 9)
        if classification_lower[index] >= -margins.delta_P and upper[index] <= margins.delta_P)
    better = tuple(index for index in range(1, 9) if upper[index] < -margins.delta_P)
    negative_relation = ("lower_bound_exceeds_margin" if np.any(neg_lower > margins.delta_P)
        else "upper_bound_exceeds_margin_without_artifact_bound" if np.any(neg_upper > margins.delta_P)
        else "all_upper_bounds_at_most_margin")
    diagnostics = []
    sufficient = upper[0] <= margins.delta_C
    if linear.lower[0] > margins.delta_C and sufficient:
        diagnostics.append("readout_insufficiency")
    if sufficient:
        diagnostics.append("feature_and_readout_sufficiency")
    if equivalent or better:
        diagnostics.append("partition_nonspecificity")
    audit_matches = []
    if role == "audit":
        if np.all(lower > thresholds):
            audit_matches.append("primary_positive_lower_bound_conjunction")
        if (np.all(np.asarray(development.primary_positive.mean) > 2 * thresholds)
                and np.any(upper <= thresholds)):
            audit_matches.append("new_participant_nonreplication")
        audit_matches.extend(diagnostics)
    with np.errstate(over="raise", invalid="raise"):
        differences = (mean-thresholds, lower-thresholds, upper-thresholds,
                       pos_lower-pos_thresholds, neg_lower-margins.delta_P, neg_upper-margins.delta_P)
    if not all(np.isfinite(value).all() for value in differences):
        raise ValueError("margin_comparison_not_numerically_representable")
    return NumericalClassification(*(tuple(map(float, value)) for value in differences),
        equivalent, better, negative_relation,
        ("all_means_exceed_margins" if np.all(mean > thresholds) else "at_least_one_mean_at_most_margin")
        if role == "development" else None, tuple(audit_matches), tuple(diagnostics))


def infer_participant_endpoints(*, role, participant_ids, primary, positive, negative,
                               linear_label, margins, development_evidence=None):
    """Compute complete fixed families; no participant filtering or terminal gate.

    Mapping keys are PRIMARY/POSITIVE/NEGATIVE; all vectors follow the supplied
    complete roster order. linear_label is one vector. For audit, pass the
    development ParticipantEvidence so the exact >2-margin nonreplication
    antecedent can be evaluated with the SAME fixed margins and disjoint people.
    The caller, not this array-only function, authenticates role assignment.
    """
    if role not in ROLE_SIZES or not isinstance(margins, Margins):
        raise ValueError("original_role_and_frozen_margins_required")
    people = tuple(participant_ids)
    if (len(people) != ROLE_SIZES[role] or any(not isinstance(person, str) or not person for person in people)
            or len(set(people)) != len(people)):
        raise ValueError("complete_unique_assigned_participant_roster_required")
    if role == "development":
        if development_evidence is not None:
            raise ValueError("development_has_no_previous_cohort_input")
    elif (not isinstance(development_evidence, ParticipantEvidence)
          or development_evidence.role != "development"
          or len(development_evidence.participant_ids) != 30
          or development_evidence.margins != margins
          or set(people) & set(development_evidence.participant_ids)):
        raise ValueError("same_margin_disjoint_complete_development_evidence_required")
    n = len(people)
    p = _vectors(primary, PRIMARY, n)
    q = _vectors(positive, POSITIVE, n)
    a = _vectors(negative, NEGATIVE, n)
    linear = _vectors({LINEAR[0]: linear_label}, LINEAR, n)
    one = _bounds(p, PRIMARY, two_sided=False)
    two = _bounds(p, PRIMARY, two_sided=True)
    pos = _bounds(q, POSITIVE, two_sided=False)
    neg = _bounds(a, NEGATIVE, two_sided=True)
    lin = _bounds(linear, LINEAR, two_sided=False)
    return ParticipantEvidence(role, people, margins, one, two, pos, neg, lin,
        _numerical(role, margins, one, two, pos, neg, lin, development_evidence))
