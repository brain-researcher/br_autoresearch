"""Prospective EP06 development diagnostics for seven paired session profiles.

This array-only module has no reader, search, final-test or opening authority.
Its 128 assignments relabel physical regional profiles within each session:
bits0..2 are the supplied Mihili order, bits3..6 the Chewie-L order; bit0 means
ordinary M1=1/PMd=0, bit1 swaps those labels. Label-independent physical profiles
stay fixed. Every distinct label-dependent source classifier is fitted again;
only its label-independent target predictions are cached across target labels.
This is eight Mihili plus sixteen Chewie source fits, not 128 new neural fits.

Inclusive upper-tail fractions describe a complete joint-label reference. They
are not selection-adjusted p-values, two directional hypothesis tests, an IUT,
or evidence about an independent animal population. Independent session-wise
exchangeability is a strong assumption not established by these calculations.
An unavailable observed profile or any unavailable shuffled classifier makes
the complete reference unavailable. Preserve partial measured values and all
128 assignment positions; never drop, redraw or impute an undefined score.

Signed agreement separately averages raw M1-minus-PMd vectors equally over
sessions within each animal and evaluates all seven session omissions. The
centered-shape diagnostic centers these two raw mean vectors over blocks;
it is not a classifier refit or an additional adopted scientific threshold.
The coordinator must record conventions before empirical use of this module.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

import profile_classifier as classifiers
from primary_profile import ProfileScore, SignedProfile


ANIMALS = ("Mihili", "Chewie-L")
ASSIGNMENTS = tuple(tuple((number >> bit) & 1 for bit in range(7)) for number in range(128))


@dataclass(frozen=True)
class ReferenceIssue:
    stage: str
    source_animal: str | None
    source_swap_bits: tuple
    reason: str


@dataclass(frozen=True)
class DirectionalReference:
    source_animal: str
    target_animal: str
    source_sessions: tuple
    target_sessions: tuple
    observed_balanced_accuracy: float | None
    assignment_balanced_accuracies: tuple
    inclusive_upper_tail_count: int | None
    inclusive_upper_tail_fraction: float | None
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True)
class JointLabelReference:
    assignments: tuple
    directions: tuple[DirectionalReference, ...]
    blocks: tuple
    source_fit_attempts: tuple
    complete: bool
    issues: tuple[ReferenceIssue, ...]


@dataclass(frozen=True)
class OmissionCosine:
    omitted_animal: str
    omitted_session: str
    cosine: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class SignedAgreement:
    blocks: tuple
    mihili_sessions: tuple
    chewie_sessions: tuple
    mihili_raw_mean: tuple | None
    chewie_raw_mean: tuple | None
    raw_cosine: float | None
    omissions: tuple[OmissionCosine, ...]
    raw_available: bool
    raw_all_positive: bool | None
    raw_unavailable_reason: str | None
    centered_shape_cosine: float | None
    centered_shape_unavailable_reason: str | None


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _panel(mihili, chewie, mihili_sessions, chewie_sessions, check):
    """Bind all metadata before inspecting the already-computed score values."""
    _check(check)
    panels, rosters = (mihili, chewie), (mihili_sessions, chewie_sessions)
    for animal, roster, count in zip(ANIMALS, rosters, (3, 4)):
        classifiers._roster(animal, roster)
        if len(roster) != count:
            raise ValueError("the complete prospective three-Mihili/four-Chewie development roster is required")
    if set(mihili_sessions) & set(chewie_sessions):
        raise ValueError("animals require distinct session identities")
    for panel, roster in zip(panels, rosters):
        if not isinstance(panel, tuple):
            raise TypeError("complete ordered profiles must be supplied as tuples")
        if len(panel) != len(roster) or any(value is None for value in panel):
            return panels, rosters, (), "required_observed_session_profile_missing"
        for value in panel:
            if not isinstance(value, classifiers.PairedSessionProfiles):
                raise TypeError("explicit animal-bound paired session profiles required")
            if value.profile is None:
                return panels, rosters, (), "required_observed_session_profile_missing"
            if not isinstance(value.profile, SignedProfile):
                raise TypeError("explicit paired regional profile required")
            if value.profile.m1 is None or value.profile.pmd is None:
                return panels, rosters, (), "required_observed_regional_profile_missing"
    blocks = classifiers._profile_metadata(mihili, ANIMALS[0], mihili_sessions)
    classifiers._profile_metadata(chewie, ANIMALS[1], chewie_sessions, blocks)
    for panel in panels:
        for supplied in panel:
            _check(check)
            for region in (supplied.profile.m1, supplied.profile.pmd):
                if not isinstance(region, ProfileScore):
                    raise TypeError("explicit regional profile required")
                for block in region.blocks:
                    if block.score is None:
                        return panels, rosters, blocks, "required_observed_profile_block_unavailable"
                    if isinstance(block.score, (bool, np.bool_)) or not isinstance(block.score, Real):
                        raise ValueError("observed profile scores must be real numbers or explicitly unavailable")
                    if not math.isfinite(float(block.score)):
                        return panels, rosters, blocks, "nonfinite_observed_profile_score"
    return panels, rosters, blocks, None


def _labels(animal, roster, bits):
    return tuple(classifiers.PairedLabels(animal, session, 1 - bit, bit)
                 for session, bit in zip(roster, bits))


def evaluate_development_reference(mihili_profiles, chewie_profiles, *,
                                   mihili_sessions, chewie_sessions, recipe, check):
    """Return all joint-label accuracy slots and inclusive diagnostic ranks.

The supplied recipe and all label-independent profile construction must be
fixed for this candidate. If upstream choices depend on permuted labels, this
primitive alone is not their complete-analysis reference. A failed null fit
stops further reference work and leaves each uncomputed slot explicitly None.
"""
    _check(check)
    recipe = classifiers._classifier_recipe(recipe)
    panels, rosters, blocks, missing = _panel(mihili_profiles, chewie_profiles,
                                            mihili_sessions, chewie_sessions, check)
    scores = [[None] * 128 for _ in ANIMALS]
    observed, fits, issues, cached = [None, None], [0, 0], [], {}

    def report(complete):
        reason = None if complete else issues[-1].reason
        directions = []
        for source in range(2):
            count = sum(value >= observed[source] for value in scores[source]) if complete else None
            directions.append(DirectionalReference(ANIMALS[source], ANIMALS[1 - source],
                rosters[source], rosters[1 - source], observed[source], tuple(scores[source]),
                count, count / 128. if complete else None, complete, reason))
        return JointLabelReference(ASSIGNMENTS, tuple(directions), blocks, tuple(fits), complete, tuple(issues))

    if missing is not None:
        issues.append(ReferenceIssue("observed_panel", None, (), missing))
        return report(False)

    def predictions(source, bits):
        key = source, bits
        if key not in cached:
            _check(check)
            fits[source] += 1
            fitted = classifiers.fit_profile_classifier(panels[source], source_animal=ANIMALS[source],
                expected_sessions=rosters[source], labels=_labels(ANIMALS[source], rosters[source], bits),
                recipe=recipe, check=check)
            cached[key] = classifiers.predict_profile_classifier(fitted, panels[1 - source],
                target_animal=ANIMALS[1 - source], expected_sessions=rosters[1 - source], check=check)
        return cached[key]

    def score(source, source_bits, target_bits):
        prediction = predictions(source, source_bits)
        return classifiers.score_profile_predictions(prediction,
            labels=_labels(ANIMALS[1 - source], rosters[1 - source], target_bits), check=check).balanced_accuracy

    # Record both ordinary directional scores before encountering a null fit.
    for source in range(2):
        bits = (0,) * len(rosters[source])
        try:
            observed[source] = scores[source][0] = score(source, bits, (0,) * len(rosters[1 - source]))
        except classifiers.ClassificationUnavailable as error:
            issues.append(ReferenceIssue("observed_classifier", ANIMALS[source], bits, str(error)))
            return report(False)
    for index, assignment in enumerate(ASSIGNMENTS[1:], 1):
        _check(check)
        labels = assignment[:3], assignment[3:]
        for source in range(2):
            try:
                scores[source][index] = score(source, labels[source], labels[1 - source])
            except classifiers.ClassificationUnavailable as error:
                issues.append(ReferenceIssue("null_classifier", ANIMALS[source], labels[source], str(error)))
                return report(False)
    return report(True)


def _mean(rows):
    # Divide before summation to avoid avoidable overflow of finite means.
    try:
        result = tuple(math.fsum(float(row[column]) / len(rows) for row in rows)
                       for column in range(len(rows[0])))
        return result if all(math.isfinite(value) for value in result) else None
    except (OverflowError, ValueError):
        return None


def _cosine(left, right):
    if left is None or right is None or not all(math.isfinite(x) for x in left + right):
        return None, "nonfinite_mean_difference"
    left_scale, right_scale = max(abs(x) for x in left), max(abs(x) for x in right)
    if left_scale == 0 or right_scale == 0:
        return None, "zero_mean_difference_norm"
    # Scaling protects a legitimate direction against norm overflow/underflow.
    a, b = tuple(x / left_scale for x in left), tuple(x / right_scale for x in right)
    denominator = math.sqrt(math.fsum(x * x for x in a) * math.fsum(x * x for x in b))
    value = math.fsum(x * y for x, y in zip(a, b)) / denominator
    return value, None


def _center(values):
    if values is None or not all(math.isfinite(value) for value in values):
        return None
    try:
        location = math.fsum(value / len(values) for value in values)
        centered = tuple(value - location for value in values)
    except (OverflowError, ValueError):
        return None
    return centered if all(math.isfinite(x) for x in centered) else None


def signed_profile_agreement(mihili_profiles, chewie_profiles, *,
                             mihili_sessions, chewie_sessions, check):
    """Raw equal-session sign agreement plus separate centered-shape cosine."""
    panels, rosters, blocks, missing = _panel(mihili_profiles, chewie_profiles,
                                            mihili_sessions, chewie_sessions, check)
    if missing is not None:
        omissions = tuple(OmissionCosine(animal, session, None, missing)
                         for animal, roster in zip(ANIMALS, rosters) for session in roster)
        return SignedAgreement(blocks, *rosters, None, None, None, omissions,
                               False, None, missing, None, missing)
    rows = []
    for panel in panels:
        animal_rows = []
        for supplied in panel:
            _check(check)
            # Recompute from bound physical score pairs, not a possibly stale
            # cached m1_minus_pmd field in the supplied SignedProfile.
            animal_rows.append(tuple(float(a.score) - float(b.score)
                for a, b in zip(supplied.profile.m1.blocks, supplied.profile.pmd.blocks)))
        rows.append(tuple(animal_rows))
    means = tuple(_mean(animal_rows) for animal_rows in rows)
    cosine, reason = _cosine(*means)
    omissions = []
    for animal_index, (animal, roster) in enumerate(zip(ANIMALS, rosters)):
        for index, session in enumerate(roster):
            _check(check)
            omitted_means = list(means)
            omitted_means[animal_index] = _mean(rows[animal_index][:index] + rows[animal_index][index + 1:])
            value, unavailable = _cosine(*omitted_means)
            omissions.append(OmissionCosine(animal, session, value, unavailable))
    shape, shape_reason = _cosine(_center(means[0]), _center(means[1]))
    reasons = tuple(value for value in (reason,) + tuple(item.unavailable_reason for item in omissions)
                    if value is not None)
    available = not reasons
    positive = bool(cosine > 0 and all(item.cosine > 0 for item in omissions)) if available else None
    return SignedAgreement(blocks, *rosters, means[0], means[1], cosine, tuple(omissions), available,
                           positive, reasons[0] if reasons else None, shape, shape_reason)
