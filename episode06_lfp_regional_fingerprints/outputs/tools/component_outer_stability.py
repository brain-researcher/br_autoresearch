"""EP06 outer stability, using recorded nominations and nominee-only outcomes.

SEARCH_POLICY.component_prediction.outer_session_assessment requires the same
nominee in all three Mihili folds or at least three of four Chewie folds,
no tied fold and no left-out M1/PMd gain-sign contradiction. A fourth
DIFFERENT untied Chewie nominee is allowed. The
left-out comparison of every nominated fold is used, including that fourth
component. No positive-gain or preservation margin is added to these selection
conditions; stable selection alone is not a supported component claim.

The caller supplies all three/four recorded NominationResults and one comparison or
explicit None per fold. None for a nominated fold means missing evaluation.
A fold without a nominee permits no comparison: its sign condition remains
unassessed, rather than treating absence as agreement. A known failed condition
establishes instability even if other evidence is unresolved; otherwise missing
evidence leaves stability unresolved. The individual conditions are retained.

This reducer reads no arrays or files, evaluates no component and changes no
stored nomination. Rechecking its recorded two/three-session arithmetic cannot
authenticate when a nomination or H prediction was frozen, nor enforce access
order. H, final directional nomination and reserved tests remain separate.
"""
from collections.abc import Mapping
from dataclasses import dataclass

import component_nomination as nomination
from component_scoring import RegionalComparison, compare_regions


class OuterStabilityError(ValueError):
    """The supplied records do not bind the permitted three/four-fold assessment."""


@dataclass(frozen=True)
class OuterFoldAssessment:
    left_out_session: str
    nomination: nomination.NominationResult
    comparison: RegionalComparison | None
    evaluation_state: str
    regional_signs: tuple[int, int] | None
    signs_agree: bool | None


@dataclass(frozen=True)
class OuterStabilityAssessment:
    target_animal: str
    target_development_sessions: tuple
    folds: tuple[OuterFoldAssessment, ...]
    nomination_counts: tuple[tuple[str, int], ...]
    repeated_component: str | None
    same_component_in_at_least_three: bool
    no_tied_fold: bool
    no_left_out_sign_contradiction: bool | None
    failed_conditions: tuple[str, ...]
    unresolved_evidence: tuple[tuple[str, str], ...]
    status: str
    stable_component: str | None
    establishes_component_claim: bool = False


def _bind_nomination(value, animal, roster, left_out):
    if (not isinstance(value, nomination.NominationResult) or value.stage != "outer"
            or value.target_animal != animal or value.target_development_sessions != roster
            or value.left_out_session != left_out):
        raise OuterStabilityError("each canonical fold needs its own recorded outer nomination")
    expected = tuple(session for session in roster if session != left_out)
    try:
        recorded = nomination._recorded_panels(value, expected)
        actual = nomination.nominate_outer_component(target_animal=animal,
            target_development_sessions=roster, left_out_session=left_out, comparisons=recorded)
    except ValueError as error:
        raise OuterStabilityError(str(error)) from error
    if value != actual:
        raise OuterStabilityError("nomination differs from its recorded prefold evidence")
    return value


def _bind_comparison(record, selected, session):
    if not isinstance(record, RegionalComparison) or record.component != selected:
        raise OuterStabilityError("only the prefold nominee's left-out comparison is permitted")
    for score in (record.m1, record.pmd):
        if score.design.session_id != session or score.design.role != "evaluation":
            raise OuterStabilityError("left-out comparison session or evaluation role changed")
        if score.status not in ("scoreable", "unscoreable"):
            raise OuterStabilityError("left-out numerical scoreability must be explicit")
    if record != compare_regions(m1=record.m1, pmd=record.pmd):
        raise OuterStabilityError("left-out regional comparison differs from its score records")
    if record.m1.status != "scoreable" or record.pmd.status != "scoreable":
        return "unscoreable_evaluation", None, None
    signs = tuple((score.regional_gain > 0) - (score.regional_gain < 0)
                  for score in (record.m1, record.pmd))
    agree = signs[0] == signs[1]
    return ("sign_agreement" if agree else "sign_contradiction"), signs, agree


def assess_outer_stability(*, target_animal, target_development_sessions,
                           nominations, left_out_comparisons):
    """Reduce all three/four outer records without opening other components.

    Both mappings have exactly the three Mihili or four Chewie canonical keys.
    Every nomination is an existing outer result, including tied or unsupported
    results. Comparisons are nominee-only RegionalComparison values or explicit
    None, never whole C/D/E panels. Bind both key sets and every nomination
    before obtaining any left-out comparison. No inference about reader access
    or the actual chronology is made from these in-memory records.
    """
    roster = nomination._roster(target_animal, target_development_sessions)
    if any(not isinstance(values, Mapping) or set(values) != set(roster)
           for values in (nominations, left_out_comparisons)):
        raise OuterStabilityError("exactly the amended canonical development fold keys are required")
    selected = {session: _bind_nomination(nominations[session], target_animal, roster, session)
                for session in roster}
    folds, unresolved = [], []
    for session in roster:
        result, record = selected[session], left_out_comparisons[session]
        if result.nominee is None:
            if record is not None:
                raise OuterStabilityError("a fold without a nominee permits no left-out component comparison")
            state, signs, agree = "no_prefold_nominee", None, None
        elif record is None:
            state, signs, agree = "missing_evaluation", None, None
        else:
            state, signs, agree = _bind_comparison(record, result.nominee, session)
        if agree is None:
            unresolved.append((session, state))
        folds.append(OuterFoldAssessment(session, result, record, state, signs, agree))

    counts = tuple((component, sum(value.nominee == component for value in selected.values()))
                   for component in nomination.COMPONENTS)
    repeated = next((component for component, count in counts if count >= 3), None)
    no_tie = all(value.status != "tied" for value in selected.values())
    no_contradiction = (False if any(fold.signs_agree is False for fold in folds)
                        else None if unresolved else True)
    failed = tuple(name for name, passes in (
        ("same_component_in_at_least_three", repeated is not None),
        ("no_tied_fold", no_tie),
        ("no_left_out_sign_contradiction", no_contradiction)) if passes is False)
    status = ("unstable_selection" if failed else
              "unresolved_outer_stability" if unresolved else "stable_selection")
    return OuterStabilityAssessment(target_animal, roster, tuple(folds), counts,
        repeated, repeated is not None, no_tie, no_contradiction, failed,
        tuple(unresolved), status, repeated if status == "stable_selection" else None)


def bind_outer_assessment(value, *, target_animal, target_development_sessions):
    """Recompute one retained assessment before final nomination uses its family.

    Every canonical fold must be present in order, including unresolved folds.
    Derived status/counts/signs cannot substitute for the actual nomination and
    nominee-only evaluation records. This establishes internal consistency,
    not source provenance, evaluation chronology or outcome-opening authority.
    """
    roster = nomination._roster(target_animal, target_development_sessions)
    if (not isinstance(value, OuterStabilityAssessment)
            or value.target_animal != target_animal
            or value.target_development_sessions != roster
            or not isinstance(value.folds, tuple) or len(value.folds) != len(roster)):
        raise OuterStabilityError("a complete recorded outer assessment for this target is required")
    for session, fold in zip(roster, value.folds):
        if not isinstance(fold, OuterFoldAssessment) or fold.left_out_session != session:
            raise OuterStabilityError("outer assessment must retain each canonical fold in order")
    actual = assess_outer_stability(target_animal=target_animal,
        target_development_sessions=roster,
        nominations={fold.left_out_session: fold.nomination for fold in value.folds},
        left_out_comparisons={fold.left_out_session: fold.comparison for fold in value.folds})
    if value != actual:
        raise OuterStabilityError("outer assessment differs from its recorded fold evidence")
    return value
