"""EP06's registered direction-balanced population-component arithmetic.

This array-only module implements component_details.md, not a source reader,
model fit, nomination, energy-gate selection, or outcome-opening authority.
The caller supplies frozen axes, design and genuine predictions on the same
whole trials. Identity checks cannot establish their training provenance.
Call score_component only for the component permitted at the current stage;
an outer held-out session permits its prefold nominee, never an eager C/D/E
panel. Calibration trials cannot be scored through this entrypoint.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


class ComponentScoringError(ValueError):
    """The supplied identities, shapes or frozen controls are inconsistent."""


def _ids(values, name, *, unique=True):
    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise ComponentScoringError(f"{name} cannot contain Boolean identities")
        if isinstance(value, str) and value:
            normalized.append(value)
        elif isinstance(value, Integral):
            normalized.append(int(value))
        elif isinstance(value, Real) and math.isfinite(float(value)):
            normalized.append(float(value))
        else:
            raise ComponentScoringError(f"{name} needs nonempty strings or finite numeric identities")
    result = tuple(normalized)
    if not result or (unique and len(set(result)) != len(result)):
        raise ComponentScoringError(f"{name} must be nonempty and have the declared unique support")
    return result


@dataclass(frozen=True)
class FrozenDesign:
    """Metadata only; order is supplied, never inferred from response values.

    Each row is one whole trial with every declared native time bin and axis.
    Axis identities bind the caller's already frozen population coordinates.
    Different regions may have different axis identities but must have equal
    dimension when their gains are compared.
    """

    session_id: str
    role: str
    directions: tuple
    trial_ids: tuple
    trial_directions: tuple
    time_ids: tuple
    axis_ids: tuple

    def __post_init__(self):
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ComponentScoringError("session_id must be an explicit nonempty string")
        if self.role not in ("training", "calibration", "evaluation"):
            raise ComponentScoringError("unknown whole-trial role")
        for name in ("directions", "trial_ids", "time_ids", "axis_ids", "trial_directions"):
            object.__setattr__(self, name, _ids(getattr(self, name), name,
                                               unique=name != "trial_directions"))
        if len(self.trial_ids) != len(self.trial_directions):
            raise ComponentScoringError("one preassigned direction is required for every whole trial")
        if set(self.trial_directions) != set(self.directions):
            raise ComponentScoringError("every declared direction must be present, with no extra directions")

    @property
    def shape(self):
        return len(self.trial_ids), len(self.time_ids), len(self.axis_ids)


@dataclass(frozen=True)
class BoundActivity:
    """Values and their explicit row/bin/coordinate identities.

    Construction does not decode values. Scoring validates all four metadata
    bindings before converting any values and owns its numerical snapshots.
    """

    design: FrozenDesign
    values: object


@dataclass(frozen=True)
class RuleScore:
    rule: str
    normalized_squared_error: float
    score: float


@dataclass(frozen=True)
class ComponentScore:
    region: str
    component: str
    design: FrozenDesign
    energy_gate: float
    status: str
    reason: str | None
    normalized_target_energy: float | None
    rules: tuple[RuleScore, ...]
    correct_minus_pooled: float | None
    correct_minus_swapped: float | None
    regional_gain: float | None


@dataclass(frozen=True)
class RegionalComparison:
    component: str
    status: str
    reason: str
    m1: ComponentScore
    pmd: ComponentScore
    equal_region_gain: float | None


def _component(name):
    if name not in ("C", "D", "E"):
        raise ComponentScoringError("exactly one explicit C, D or E component is required")


def _bound(design, activity):
    if not isinstance(design, FrozenDesign) or not isinstance(activity, BoundActivity):
        raise ComponentScoringError("an explicit frozen design and bound activity are required")
    if not isinstance(activity.design, FrozenDesign) or activity.design != design:
        raise ComponentScoringError("trial, direction, time, axis, session or role identities differ")


def _snapshot(design, activity):
    try:
        raw = np.asarray(activity.values)
        if np.iscomplexobj(raw):
            raise ComponentScoringError("population activity must be real, not complex")
        value = np.array(raw, dtype=np.float64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ComponentScoringError("activity must be a numeric whole-trial array") from exc
    if value.shape != design.shape:
        raise ComponentScoringError(f"activity shape must be {design.shape}")
    return value


def _groups(design):
    return tuple(np.asarray([i for i, label in enumerate(design.trial_directions)
                             if label == direction], dtype=np.intp)
                 for direction in design.directions)


def _project(value, groups, component):
    # Group means preserve each direction's total weight when counts differ.
    means = [np.mean(value[rows], axis=0) for rows in groups]
    result = np.empty_like(value)
    if component == "E":
        for rows, mean in zip(groups, means):
            result[rows] = value[rows] - mean
    else:
        common = np.mean(means, axis=0)
        for rows, mean in zip(groups, means):
            result[rows] = common if component == "C" else mean - common
    return result


def _energy(value, groups):
    # Mean over directions, trials and time; sum (not mean) over axes.
    return float(np.mean([np.mean(np.sum(np.square(value[rows]), axis=2))
                          for rows in groups]))


def project_component(design, activity, *, component):
    """Apply one fixed design projector; never fit axes or subtract extra means.

    This lower-level algebra requires finite values. Use score_component for
    contract-defined unresolved numerical/energy outcomes. The returned array
    is independent of the supplied activity and has exactly its frozen order.
    """
    _component(component)
    _bound(design, activity)
    value = _snapshot(design, activity)
    if not np.all(np.isfinite(value)):
        raise ComponentScoringError("projection input is nonfinite")
    with np.errstate(over="ignore", invalid="ignore"):
        projected = _project(value, _groups(design), component)
    if not np.all(np.isfinite(projected)):
        raise ComponentScoringError("projection arithmetic is nonfinite")
    return projected


def score_component(design, *, region, component, energy_gate,
                    observed, correct, pooled, swapped):
    """Score one selected component on identical evaluation trials.

    energy_gate is a caller-frozen positive normalized energy, with no default
    or adaptive estimate. Exactly-at-gate energy is scoreable. Each prediction
    is projected from its own values; observed means never enter predictions.
    Nonfinite values/arithmetic and below-gate energy return unscoreable, with
    absent scores/gains rather than zero or a failed scientific result.
    """
    _component(component)
    if region not in ("M1", "PMd"):
        raise ComponentScoringError("region must be M1 or PMd")
    if (isinstance(energy_gate, (bool, np.bool_)) or not isinstance(energy_gate, Real)
            or not math.isfinite(float(energy_gate)) or energy_gate <= 0):
        raise ComponentScoringError("an explicit finite positive energy gate is required")
    energy_gate = float(energy_gate)
    activities = (observed, correct, pooled, swapped)
    for activity in activities:
        _bound(design, activity)
    if design.role != "evaluation":
        raise ComponentScoringError("component scoring requires the preassigned evaluation role")

    def unscoreable(reason, energy=None):
        return ComponentScore(region, component, design, energy_gate, "unscoreable", reason,
                              energy, (), None, None, None)

    values = [_snapshot(design, activity) for activity in activities]
    for name, value in zip(("observed", "correct", "pooled", "swapped"), values):
        if not np.all(np.isfinite(value)):
            return unscoreable(f"nonfinite_{name}")
    groups = _groups(design)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        target = _project(values[0], groups, component)
        energy = _energy(target, groups)
        if not math.isfinite(energy):
            return unscoreable("nonfinite_target_energy")
        if energy < energy_gate:
            return unscoreable("target_energy_below_gate", energy)
        rules = []
        for name, prediction in zip(("correct", "pooled", "swapped"), values[1:]):
            predicted = _project(prediction, groups, component)
            error = _energy(target - predicted, groups)
            score = 1.0 - error / energy
            if not math.isfinite(error) or not math.isfinite(score):
                return unscoreable(f"nonfinite_{name}_score", energy)
            rules.append(RuleScore(name, error, score))
    a, b, c = (rule.score for rule in rules)
    contrasts = a - b, a - c
    gain = a - max(b, c)
    if not all(math.isfinite(value) for value in (*contrasts, gain)):
        return unscoreable("nonfinite_gain", energy)
    return ComponentScore(region, component, design, energy_gate, "scoreable", None,
                          energy, tuple(rules), *contrasts, gain)


def compare_regions(*, m1, pmd):
    """Apply the fixed regional sign and 0.01 preservation rules.

    No pooling of regional denominators or gain averaging across unequal signs
    is allowed. Both nonpositive gains mean failure, including a zero/negative
    pair; that pair has no averaged gain because its mathematical signs differ.
    This is a session/component comparison, not nomination or confirmation.
    """
    if not isinstance(m1, ComponentScore) or not isinstance(pmd, ComponentScore):
        raise ComponentScoringError("two component-score results are required")
    if m1.region != "M1" or pmd.region != "PMd" or m1.component != pmd.component:
        raise ComponentScoringError("matching selected M1 and PMd components are required")
    for field in ("session_id", "role", "directions", "trial_ids", "trial_directions", "time_ids"):
        if getattr(m1.design, field) != getattr(pmd.design, field):
            raise ComponentScoringError("regional evaluation support or whole-trial identities differ")
    if len(m1.design.axis_ids) != len(pmd.design.axis_ids):
        raise ComponentScoringError("regional population dimensions must match")
    if m1.status != "scoreable" or pmd.status != "scoreable":
        return RegionalComparison(m1.component, "unresolved", "unscoreable_region", m1, pmd, None)
    a, b = m1.regional_gain, pmd.regional_gain
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        raise ComponentScoringError("scoreable regional results require finite gains")
    sign_a, sign_b = (a > 0) - (a < 0), (b > 0) - (b < 0)
    mean = a / 2.0 + b / 2.0 if sign_a == sign_b else None
    if a <= 0 and b <= 0:
        status, reason = "failed", "both_regional_gains_nonpositive"
    elif sign_a != sign_b:
        status, reason = "unresolved", "regional_gain_signs_disagree"
    elif mean >= 0.01:
        status, reason = "preserved", "positive_regional_gains_and_fixed_margin"
    else:
        status, reason = "unresolved", "positive_mean_below_fixed_margin"
    return RegionalComparison(m1.component, status, reason, m1, pmd, mean)
