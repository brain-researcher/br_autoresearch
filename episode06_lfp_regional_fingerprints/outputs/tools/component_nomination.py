"""EP06's fixed nomination rule over already permitted component-score records.

There is no response reader, projector call, model fit or eager evaluation here.
Outer nomination receives only the other two Mihili or three Chewie target
development sessions. Final nomination receives all three/four only after its
retained outer assessment is bound and recomputed. Its final family must match
the stable outer family. Record arithmetic cannot establish the chronology or
isolation of an upstream evaluator.

Negative and mixed-sign measurements remain visible. Numerical scoreability
and candidate-specific sign eligibility are distinct, as amended on2026-10-02.
This module does not select an energy gate, fit support H or open reserved
sessions. Final directional agreement, including outer stability, is still not
a supported component claim.
"""
from collections.abc import Mapping
from dataclasses import dataclass, replace
from statistics import median

from component_scoring import RegionalComparison, compare_regions


COMPONENTS = ("C", "D", "E")
NOMINATION_MARGIN = 0.01
TARGET_SESSIONS = {
    "Mihili": ("Mihili_CO_20140217.mat", "Mihili_CO_20140218.mat",
               "Mihili_CO_20140304.mat"),
    "Chewie-L": ("Chewie_CO_20160929.mat", "Chewie_CO_20161005.mat",
                 "Chewie_CO_20161007.mat", "Chewie_CO_20161014.mat"),
}


class ComponentNominationError(ValueError):
    """Input records do not match the permitted session/component comparison."""


@dataclass(frozen=True)
class CandidateNomination:
    component: str
    eligible: bool
    reason: str
    session_equal_region_gains: tuple
    median_gain: float | None


@dataclass(frozen=True)
class NominationResult:
    stage: str
    target_animal: str
    target_development_sessions: tuple
    left_out_session: str | None
    supplied_sessions: tuple
    ranking_sessions: tuple
    unscoreable_sessions: tuple
    status: str
    nominee: str | None
    tied_components: tuple
    candidates: tuple[CandidateNomination, ...]
    records: tuple
    ranking_nominee: str | None
    outer_assessment: object | None = None
    outer_stability_evaluated: bool = False


@dataclass(frozen=True)
class DirectionalAgreement:
    status: str
    component: str | None
    mihili_to_chewie: NominationResult
    chewie_to_mihili: NominationResult
    establishes_component_claim: bool = False


def _roster(target_animal, target_development_sessions):
    if target_animal not in TARGET_SESSIONS:
        raise ComponentNominationError("target must be Mihili or Chewie-L")
    if isinstance(target_development_sessions, str):
        raise ComponentNominationError("the declared development session identities are required")
    supplied = tuple(target_development_sessions)
    canonical = TARGET_SESSIONS[target_animal]
    if len(supplied) != len(canonical) or set(supplied) != set(canonical):
        raise ComponentNominationError("target roster must contain exactly its three Mihili or four Chewie development sessions")
    return canonical


def _panels(comparisons, expected_sessions):
    # Reject an unauthorized session key before obtaining any component record.
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(expected_sessions):
        raise ComponentNominationError("only the exact permitted nomination sessions may be supplied")
    result = {}
    for session in expected_sessions:
        panel = comparisons[session]
        if not isinstance(panel, Mapping) or set(panel) != set(COMPONENTS):
            raise ComponentNominationError("every supplied session requires exactly C, D and E records")
        checked = {}
        for component in COMPONENTS:
            record = panel[component]
            if not isinstance(record, RegionalComparison) or record.component != component:
                raise ComponentNominationError("one matching existing RegionalComparison is required")
            for score in (record.m1, record.pmd):
                if score.design.session_id != session or score.design.role != "evaluation":
                    raise ComponentNominationError("score record crosses its session or evaluation role")
                if score.status not in ("scoreable", "unscoreable"):
                    raise ComponentNominationError("component scoreability must be explicit")
            # Reuse the existing sign/identity arithmetic, including zero versus
            # negative disagreement; a claimed average cannot override it.
            actual = compare_regions(m1=record.m1, pmd=record.pmd)
            if record != actual:
                raise ComponentNominationError("regional comparison differs from its component-score records")
            if checked:
                first = checked[COMPONENTS[0]]
                if record.m1.design != first.m1.design or record.pmd.design != first.pmd.design:
                    raise ComponentNominationError("all components require the same frozen evaluation design")
            checked[component] = record
        result[session] = checked
    return result


def _decide(*, stage, target_animal, roster, left_out_session, panels,
            supplied_sessions):
    common = tuple(session for session in supplied_sessions if all(
        panels[session][component].m1.status == "scoreable"
        and panels[session][component].pmd.status == "scoreable" for component in COMPONENTS))
    unscoreable = tuple(session for session in supplied_sessions if session not in common)
    enough = len(common) == len(supplied_sessions) if stage == "outer" else len(common) >= 3
    ranking = common if enough else ()
    candidates = []
    for component in COMPONENTS:
        gains = tuple((session, panels[session][component].equal_region_gain) for session in ranking)
        if not enough:
            reason, eligible, value = "complete_comparison_numerical_support_unresolved", False, None
        elif any(value is None for _, value in gains):
            reason, eligible, value = "regional_sign_disagreement", False, None
        else:
            reason, eligible = "sign_consistent_on_common_sessions", True
            value = float(median(gain for _, gain in gains))
        candidates.append(CandidateNomination(component, eligible, reason, gains, value))
    eligible = [candidate for candidate in candidates if candidate.eligible]
    nominee, tied = None, ()
    if not enough:
        status = "unresolved_numerical_support"
    elif not eligible:
        status = "no_eligible_candidate"
    else:
        best = max(candidate.median_gain for candidate in eligible)
        top = tuple(candidate.component for candidate in eligible
                    if best <= candidate.median_gain + NOMINATION_MARGIN)
        if len(top) > 1:
            status, tied = "tied", top
        else:
            status, nominee = "nominated", top[0]
    return NominationResult(stage, target_animal, roster, left_out_session,
        supplied_sessions, ranking, unscoreable, status, nominee, tied,
        tuple(candidates), tuple((session, tuple(panels[session][component] for component in COMPONENTS))
                                for session in supplied_sessions), nominee)


def _recorded_panels(value, expected):
    """Reconstruct an exact retained panel without dropping or reordering records."""
    if (value.supplied_sessions != expected or not isinstance(value.records, tuple)
            or len(value.records) != len(expected)):
        raise ComponentNominationError("nomination must retain exactly its permitted session panels")
    recorded = {}
    for session, entry in zip(expected, value.records):
        if (not isinstance(entry, tuple) or len(entry) != 2 or entry[0] != session
                or not isinstance(entry[1], tuple) or len(entry[1]) != len(COMPONENTS)):
            raise ComponentNominationError("nomination source-record order changed")
        if any(not isinstance(record, RegionalComparison) or record.component != component
               for record, component in zip(entry[1], COMPONENTS)):
            raise ComponentNominationError("nomination source components changed")
        recorded[session] = dict(zip(COMPONENTS, entry[1]))
    return recorded


def _bound_outer_assessment(value, animal, roster):
    # Stability depends on outer nomination; a local import keeps that one-way
    # module initialization while allowing final nomination to bind the result.
    from component_outer_stability import bind_outer_assessment
    try:
        return bind_outer_assessment(value, target_animal=animal,
                                     target_development_sessions=roster)
    except ValueError as error:
        raise ComponentNominationError(str(error)) from error


def nominate_outer_component(*, target_animal, target_development_sessions,
                             left_out_session, comparisons):
    """Nominate before scoring the held-out target session's one chosen component.

    ``comparisons`` is session→component→RegionalComparison for exactly the
    OTHER two Mihili or three Chewie development sessions. Any unscoreable region
    for any rival makes the complete nomination unresolved. A measurable
    mixed-sign rival is retained and excluded only from candidate eligibility.
    """
    roster = _roster(target_animal, target_development_sessions)
    if left_out_session not in roster:
        raise ComponentNominationError("left-out identity must belong to the fixed target development roster")
    expected = tuple(session for session in roster if session != left_out_session)
    panels = _panels(comparisons, expected)
    return _decide(stage="outer", target_animal=target_animal, roster=roster,
        left_out_session=left_out_session, panels=panels, supplied_sessions=expected)


def nominate_final_component(*, target_animal, target_development_sessions,
                             outer_assessment, comparisons):
    """Bind recorded outer evidence, then rank all retained target sessions.

    ``outer_assessment`` is an OuterStabilityAssessment with every canonical
    fold and its retained evidence; it is recomputed before final panels are
    accessed. Ranking requires all three common scoreable Mihili sessions or
    at least three of four Chewie sessions, using every common session. A raw
    winner is retained as ranking_nominee, but becomes the final nominee only
    if its direction's stable outer family matches. Recorded missing/unstable
    evidence remains reportable; it cannot support a reserved H prediction.
    This binding cannot prove that prospective records preceded evaluation.
    """
    roster = _roster(target_animal, target_development_sessions)
    assessment = _bound_outer_assessment(outer_assessment, target_animal, roster)
    panels = _panels(comparisons, roster)
    result = _decide(stage="final", target_animal=target_animal, roster=roster,
        left_out_session=None, panels=panels, supplied_sessions=roster)
    status, nominee = result.status, result.nominee
    if nominee is not None:
        if assessment.status == "unstable_selection":
            status, nominee = "unstable_outer_selection", None
        elif assessment.status != "stable_selection":
            status, nominee = "unresolved_outer_stability", None
        elif nominee != assessment.stable_component:
            status, nominee = "outer_family_mismatch", None
    return replace(result, status=status, nominee=nominee,
                   outer_assessment=assessment, outer_stability_evaluated=True)


def _bind_final_nomination(value, animal):
    if (not isinstance(value, NominationResult) or value.stage != "final"
            or value.target_animal != animal or value.left_out_session is not None):
        raise ComponentNominationError("both correctly directed final nomination results are required")
    roster = _roster(animal, value.target_development_sessions)
    assessment = _bound_outer_assessment(value.outer_assessment, animal, roster)
    actual = nominate_final_component(target_animal=animal,
        target_development_sessions=roster, outer_assessment=assessment,
        comparisons=_recorded_panels(value, roster))
    if value != actual:
        raise ComponentNominationError("final nomination differs from its recorded panels or outer evidence")
    return value


def compare_final_nominations(*, mihili_to_chewie, chewie_to_mihili):
    """Recompute both final records and require the same stable-bound nominee.

    Support predictions, reserved evaluation and their claim criteria remain
    separate required evidence even when this returns agreement.
    """
    for value, animal in ((mihili_to_chewie, "Chewie-L"), (chewie_to_mihili, "Mihili")):
        _bind_final_nomination(value, animal)
    if any(value.status != "nominated" for value in (mihili_to_chewie, chewie_to_mihili)):
        status, component = "unresolved_nomination", None
    elif mihili_to_chewie.nominee != chewie_to_mihili.nominee:
        status, component = "directional_disagreement", None
    else:
        status, component = "same_untied_final_nominee", mihili_to_chewie.nominee
    return DirectionalAgreement(status, component, mihili_to_chewie, chewie_to_mihili)
