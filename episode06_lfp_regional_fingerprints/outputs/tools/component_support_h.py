"""EP06's fixed H arithmetic over supplied calibration-only session summaries.

This module does not construct a profile, choose a frequency combiner, read a
source, or score evaluation trials. The raw-profile construction, combination
of regional reliability/distance into session scalars, and estimator of the
bandwise between-session distance scales remain unresolved upstream choices.
Supplying scalars here does not resolve or authenticate those choices. Target
and reference distances must be comparable under the applicable prospective
prototype/scale construction, including its other-six/all-seven scope.
Binding scalar-record keys cannot prove that upstream prototypes or scales
excluded the predicted development session or any reserved outcomes.

An outer prediction uses exactly the other six development sessions; a
reserved prediction uses all seven. References retain pooled session weights,
not animal-weighted medians. No missing session is dropped or imputed.
Only the existing nominee is admitted. Reserved predictions require two final
nominations bound to their actual stable outer families. The regional outcome
gates, reserved outcome counts/uncertainty and the EP07 freeze/opening
dependency remain separate. A returned prediction establishes no claim or
authority to open outcomes, and cannot prove it was frozen before evaluation.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from statistics import median

import component_nomination as nomination


DEVELOPMENT_SESSIONS = tuple(session for animal in ("Mihili", "Chewie-L")
                             for session in nomination.TARGET_SESSIONS[animal])
RESERVED_SESSIONS = {
    "Mihili": ("Mihili_CO_20140303.mat", "Mihili_CO_20140307.mat"),
    "Chewie-L": ("Chewie_CO_20161006.mat", "Chewie_CO_20161021.mat"),
}
MAD_FACTOR = 1.4826
H_THRESHOLD = 0.25


class SupportPredictionError(ValueError):
    """Supplied identities or nomination records do not bind this prediction."""


@dataclass(frozen=True)
class CalibrationSupport:
    """One already-defined session summary, never evaluation-derived values.

    Reliability is the supplied Spearman--Brown-corrected profile correlation;
    finite negative reliability is retained, not clipped. Distance is the
    supplied nonnegative bandwise RMS distance from the applicable source
    regional prototype. ``bandwise_distance_scale_valid=True`` means the
    caller found every required bandwise scale finite and strictly positive;
    False/None or a missing scalar requires abstention. This declaration does
    not choose a scale estimator or authenticate an upstream computation.
    """

    session_id: str
    component: str
    role: str
    reliability: object
    profile_distance: object
    bandwise_distance_scale_valid: bool | None


@dataclass(frozen=True)
class SupportPrediction:
    stage: str
    target_animal: str
    target_session: str
    component: str | None
    reference_sessions: tuple[str, ...]
    status: str
    prediction: str | None
    reasons: tuple[str, ...]
    missing_reference_sessions: tuple[str, ...] = ()
    reliability_median: float | None = None
    reliability_scale: float | None = None
    distance_median: float | None = None
    distance_scale: float | None = None
    standardized_reliability: float | None = None
    standardized_distance: float | None = None
    support_index: float | None = None
    establishes_component_claim: bool = False


def _selected(value):
    """Bind nomination identity only; do not reopen its evaluation records."""
    if value.status == "nominated":
        if value.nominee not in nomination.COMPONENTS or value.tied_components:
            raise SupportPredictionError("a nominated record requires exactly one untied component")
        return value.nominee
    if value.status not in ("unresolved_numerical_support", "no_eligible_candidate", "tied"):
        raise SupportPredictionError("unknown nomination state")
    if value.nominee is not None:
        raise SupportPredictionError("an unresolved nomination cannot supply a selected component")
    return None


def _record(value, session, component):
    if (not isinstance(value, CalibrationSupport) or value.session_id != session
            or value.component != component or value.role != "calibration"):
        raise SupportPredictionError("each input must bind its session, nominee and calibration role")
    return value


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _predict(*, stage, animal, session, component, target, references):
    expected = tuple(key for key in DEVELOPMENT_SESSIONS if stage == "reserved" or key != session)
    base = dict(stage=stage, target_animal=animal, target_session=session,
                component=component, reference_sessions=expected)
    if component is None:
        # A tied/unsupported nomination grants no reason to inspect H inputs.
        return SupportPrediction(**base, status="no_nominee", prediction=None,
                                 reasons=("no_valid_prefold_or_final_nominee",))

    if references is None:
        keys = set()
    elif not isinstance(references, Mapping):
        raise SupportPredictionError("reference inputs must map canonical sessions to calibration records")
    else:
        keys = set(references)
    if keys - set(expected):
        raise SupportPredictionError("reference inputs include a target, reserved or otherwise forbidden session")
    missing = tuple(key for key in expected if key not in keys)
    if missing:
        return SupportPrediction(**base, status="abstained", prediction=None,
            reasons=("incomplete_reference_sessions",), missing_reference_sessions=missing)

    # Bind every present record before accessing ANY numerical scalar.
    bound = {}
    missing_values = []
    for key in expected:
        value = references[key]
        if value is None:
            missing_values.append(key)
        else:
            bound[key] = _record(value, key, component)
    if target is not None:
        target = _record(target, session, component)
    if missing_values or target is None:
        reasons = (("missing_target_calibration",) if target is None else ())
        if missing_values:
            reasons += ("incomplete_reference_sessions",)
        return SupportPrediction(**base, status="abstained", prediction=None,
            reasons=reasons, missing_reference_sessions=tuple(missing_values))

    reasons, numeric = [], {}
    for key, value in ((session, target), *((key, bound[key]) for key in expected)):
        reliability = _number(value.reliability)
        distance = _number(value.profile_distance)
        if reliability is None:
            reasons.append(key + ":missing_or_nonfinite_reliability")
        if distance is None or distance < 0:
            reasons.append(key + ":missing_nonfinite_or_negative_distance")
        if value.bandwise_distance_scale_valid is not True:
            reasons.append(key + ":bandwise_distance_scale_unresolved")
        numeric[key] = reliability, distance
    if reasons:
        return SupportPrediction(**base, status="abstained", prediction=None, reasons=tuple(reasons))

    reliability_median = float(median(numeric[key][0] for key in expected))
    distance_median = float(median(numeric[key][1] for key in expected))
    reliability_scale = MAD_FACTOR * median(abs(numeric[key][0] - reliability_median) for key in expected)
    distance_scale = MAD_FACTOR * median(abs(numeric[key][1] - distance_median) for key in expected)
    summaries = dict(reliability_median=reliability_median, reliability_scale=reliability_scale,
                     distance_median=distance_median, distance_scale=distance_scale)
    for name, value in summaries.items():
        if not math.isfinite(value) or (name.endswith("scale") and value <= 0):
            reasons.append("zero_or_nonfinite_reference_" + name)
    if reasons:
        # Do not expose nonfinite output values as usable calibration state.
        summaries = {name: value if math.isfinite(value) else None for name, value in summaries.items()}
        return SupportPrediction(**base, status="abstained", prediction=None,
                                 reasons=tuple(reasons), **summaries)
    z_reliability = (numeric[session][0] - reliability_median) / reliability_scale
    z_distance = (numeric[session][1] - distance_median) / distance_scale
    support = z_reliability - z_distance
    if not all(math.isfinite(value) for value in (z_reliability, z_distance, support)):
        return SupportPrediction(**base, status="abstained", prediction=None,
                                 reasons=("nonfinite_standardization_or_H",), **summaries)
    prediction = "preserved" if support >= H_THRESHOLD else "failed" if support <= -H_THRESHOLD else None
    return SupportPrediction(**base, status="predicted" if prediction else "abstained",
        prediction=prediction, reasons=() if prediction else ("H_inside_abstention_interval",),
        standardized_reliability=z_reliability, standardized_distance=z_distance,
        support_index=support, **summaries)


def predict_outer_support(*, nomination_result, target, references):
    """Predict the recorded outer nominee from the OTHER six sessions only.

    ``target`` and each reference are CalibrationSupport values (or explicit
    None). No left-out evaluation comparison is accepted or inspected. Missing
    reference keys abstain; extra keys are rejected before obtaining values.
    """
    value = nomination_result
    if not isinstance(value, nomination.NominationResult) or value.stage != "outer":
        raise SupportPredictionError("an existing outer NominationResult is required")
    animal = value.target_animal
    if animal not in nomination.TARGET_SESSIONS:
        raise SupportPredictionError("unknown target animal")
    roster = nomination.TARGET_SESSIONS[animal]
    session = value.left_out_session
    if (value.target_development_sessions != roster or session not in roster
            or value.supplied_sessions != tuple(key for key in roster if key != session)):
        raise SupportPredictionError("outer nomination does not bind its canonical left-out session")
    return _predict(stage="development", animal=animal, session=session,
                    component=_selected(value), target=target, references=references)


def predict_reserved_support(*, agreement, target_session, target, references):
    """Predict one canonical reserved session using all seven development ones.

    Both final panels and their retained outer assessments are recomputed
    before any calibration scalar is inspected. Only the same untied final
    nominee matching each direction's stable family admits H inputs; valid
    unresolved/disagreeing records return no_nominee. All four predictions
    still must precede reserved evaluation; these records cannot authenticate
    chronology or override the EP07 freeze/opening prerequisite.
    """
    if not isinstance(agreement, nomination.DirectionalAgreement):
        raise SupportPredictionError("an existing final DirectionalAgreement is required")
    animal = next((name for name, sessions in RESERVED_SESSIONS.items() if target_session in sessions), None)
    if animal is None:
        raise SupportPredictionError("target must be one of the four canonical reserved sessions")
    try:
        actual = nomination.compare_final_nominations(
            mihili_to_chewie=agreement.mihili_to_chewie,
            chewie_to_mihili=agreement.chewie_to_mihili)
    except ValueError as error:
        raise SupportPredictionError(str(error)) from error
    if agreement != actual:
        raise SupportPredictionError("final agreement differs from its retained final and outer evidence")
    return _predict(stage="reserved", animal=animal, session=target_session,
                    component=actual.component, target=target, references=references)
