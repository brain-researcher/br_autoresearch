"""Prospective EP06 source-scaled signed magnitude classifiers.

This array-only branch consumes already-computed, explicitly paired physical
profiles. For each regional row, m is the signed mean of its K scores. Source
rows alone determine c=median(m) and q=1.4826*median(abs(m-c)). A nonpositive or
nonfinite q makes the complete comparison unavailable; there is no epsilon,
replacement scale, clipping, target fitting or row omission.

For K>1 the feature vector is [h, (m-c)/q], where h is the unchanged centered
unit-L2 profile shape. The two blocks have fixed 1:1 weights and the combined
vector is NOT normalized again. K=1 instead uses [(s-c)/q] explicitly, without
calling or repairing the undefined singleton shape. Logistic lambda is fixed
at 1, with the existing mean-logloss objective and unpenalized intercept.

The complete 128-assignment diagnostic refits each distinct source-label
classifier (8+16 fits). Its inclusive upper-tail fractions are descriptive,
not selection-adjusted p-values or independent-animal inference. Neural
fitting, search decisions and outcome-opening authority remain with callers.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from bandwise_prediction import FeatureBlock
import profile_classifier as classifiers
import profile_references as references


@dataclass(frozen=True)
class FittedMagnitudeClassifier:
    source_animal: str
    source_sessions: tuple[str, ...]
    source_labels: tuple[classifiers.PairedLabels, ...]
    blocks: tuple[FeatureBlock, ...]
    recipe: classifiers.ClassifierRecipe
    representation: str
    magnitude_center: float
    magnitude_scale: float
    coefficients: np.ndarray
    intercept: float
    objective: float
    iterations: int
    maximum_absolute_gradient: float


def _recipe(recipe):
    recipe = classifiers._classifier_recipe(recipe)
    if recipe.method != "l2_logistic" or recipe.penalty != 1.:
        raise ValueError("signed magnitude requires the fixed l2_logistic lambda=1 recipe")
    return recipe


def _representation(blocks):
    return "singleton_signed_magnitude" if len(blocks) == 1 else "centered_unit_l2_shape_plus_signed_magnitude"


def _means(profiles, check):
    means = []
    for supplied in profiles:
        for profile in (supplied.profile.m1, supplied.profile.pmd):
            classifiers._check(check)
            values = tuple(block.score for block in profile.blocks)
            if any(value is None for value in values):
                raise classifiers.ClassificationUnavailable("required_profile_block_unavailable")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   for value in values):
                raise ValueError("profile scores must be explicit real numbers or unavailable")
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.asarray(values, dtype=np.float64)
                mean = float(np.mean(vector))
            if not np.isfinite(vector).all() or not math.isfinite(mean):
                raise classifiers.ClassificationUnavailable("nonfinite_signed_profile_magnitude")
            means.append(mean)
    return np.asarray(means, dtype=np.float64)


def _matrix(profiles, blocks, means, center, scale, check):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        magnitude = (means - center) / scale
    if not np.isfinite(magnitude).all():
        raise classifiers.ClassificationUnavailable("nonfinite_source_scaled_magnitude")
    if len(blocks) == 1:
        return magnitude[:, None]
    # Reuse exactly the old shape convention, without modifying old classes
    # or silently rescuing a zero/nonfinite multi-band centered profile.
    shape = classifiers._shape_matrix(profiles, check)
    return np.column_stack((shape, magnitude))


def fit_magnitude_classifier(profiles, *, source_animal, expected_sessions,
                             labels, recipe, check):
    """Fit source-only scale and the explicit fixed-lambda classifier."""
    classifiers._check(check)
    sessions = classifiers._roster(source_animal, expected_sessions)
    recipe = _recipe(recipe)
    y = classifiers._labels(labels, source_animal, sessions)
    blocks = classifiers._profile_metadata(profiles, source_animal, sessions)
    means = _means(profiles, check)
    with np.errstate(over="ignore", invalid="ignore"):
        center = float(np.median(means))
        scale = float(1.4826 * np.median(np.abs(means - center)))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise classifiers.ClassificationUnavailable("zero_or_nonfinite_source_magnitude_MAD")
    x = _matrix(profiles, blocks, means, center, scale, check)
    parameters, objective, iterations, gradient = classifiers._fit_logistic(x, y, 1., check)
    classifiers._check(check)
    return FittedMagnitudeClassifier(source_animal, sessions, labels, blocks, recipe,
        _representation(blocks), center, scale, classifiers._readonly(parameters[:-1]),
        float(parameters[-1]), objective, iterations, gradient)


def predict_magnitude_classifier(fitted, profiles, *, target_animal,
                                 expected_sessions, check):
    """Apply the frozen source scale and classifier, without target labels."""
    classifiers._check(check)
    if not isinstance(fitted, FittedMagnitudeClassifier):
        raise TypeError("frozen source-animal magnitude classifier required")
    classifiers._roster(fitted.source_animal, fitted.source_sessions)
    classifiers._labels(fitted.source_labels, fitted.source_animal, fitted.source_sessions)
    _recipe(fitted.recipe)
    sessions = classifiers._roster(target_animal, expected_sessions)
    if target_animal == fitted.source_animal or set(sessions) & set(fitted.source_sessions):
        raise ValueError("cross-animal prediction requires a different animal and disjoint session identities")
    blocks = classifiers._profile_metadata(profiles, target_animal, sessions, fitted.blocks)
    if (fitted.representation != _representation(blocks)
            or not math.isfinite(fitted.magnitude_center)
            or not math.isfinite(fitted.magnitude_scale) or fitted.magnitude_scale <= 0):
        raise ValueError("frozen magnitude representation requires its finite positive source MAD scale")
    dimension = 1 if len(blocks) == 1 else len(blocks) + 1
    coefficients = np.asarray(fitted.coefficients)
    if (coefficients.dtype.kind != "f" or coefficients.shape != (dimension,)
            or not np.isfinite(coefficients).all() or not math.isfinite(fitted.intercept)
            or not math.isfinite(fitted.maximum_absolute_gradient)
            or fitted.maximum_absolute_gradient > 1e-9):
        raise ValueError("frozen magnitude classifier must retain its finite converged source fit")
    means = _means(profiles, check)
    x = _matrix(profiles, blocks, means, fitted.magnitude_center, fitted.magnitude_scale, check)
    with np.errstate(over="ignore", invalid="ignore"):
        margins = x @ coefficients + fitted.intercept
    if not np.isfinite(margins).all():
        raise classifiers.ClassificationUnavailable("nonfinite_classifier_decision_score")
    rows = tuple(classifiers.PredictionRow(session, region, float(margin),
                         None if margin == 0 else int(margin > 0))
                 for (session, region), margin in zip(
                     ((session, region) for session in sessions for region in ("M1", "PMd")), margins))
    classifiers._check(check)
    return classifiers.ProfilePredictions(fitted.source_animal, fitted.source_sessions,
                                          target_animal, sessions, blocks, rows)


def evaluate_magnitude_reference(mihili_profiles, chewie_profiles, *,
                                 mihili_sessions, chewie_sessions, recipe, check):
    """Complete joint-label reference using this branch's source refits only.

    The orchestration mirrors the frozen profile reference but calls explicit
    magnitude fit/predict functions locally; no shared function is patched.
    Any unavailable observed or relabeled fit stops the entire reference,
    retaining partial values and all 128 slots without a partial tail rank.
    """
    classifiers._check(check)
    recipe = _recipe(recipe)
    panels, rosters, blocks, missing = references._panel(mihili_profiles, chewie_profiles,
                                                        mihili_sessions, chewie_sessions, check)
    scores = [[None] * 128 for _ in references.ANIMALS]
    observed, fits, issues, cached = [None, None], [0, 0], [], {}

    def report(complete):
        reason = None if complete else issues[-1].reason
        directions = []
        for source in range(2):
            count = sum(value >= observed[source] for value in scores[source]) if complete else None
            directions.append(references.DirectionalReference(
                references.ANIMALS[source], references.ANIMALS[1 - source],
                rosters[source], rosters[1 - source], observed[source], tuple(scores[source]),
                count, count / 128. if complete else None, complete, reason))
        return references.JointLabelReference(references.ASSIGNMENTS, tuple(directions), blocks,
                                              tuple(fits), complete, tuple(issues))

    if missing is not None:
        issues.append(references.ReferenceIssue("observed_panel", None, (), missing))
        return report(False)

    def predictions(source, bits):
        key = source, bits
        if key not in cached:
            classifiers._check(check)
            fits[source] += 1
            fitted = fit_magnitude_classifier(panels[source], source_animal=references.ANIMALS[source],
                expected_sessions=rosters[source],
                labels=references._labels(references.ANIMALS[source], rosters[source], bits),
                recipe=recipe, check=check)
            cached[key] = predict_magnitude_classifier(fitted, panels[1 - source],
                target_animal=references.ANIMALS[1 - source],
                expected_sessions=rosters[1 - source], check=check)
        return cached[key]

    def score(source, source_bits, target_bits):
        prediction = predictions(source, source_bits)
        return classifiers.score_profile_predictions(prediction,
            labels=references._labels(references.ANIMALS[1 - source], rosters[1 - source], target_bits),
            check=check).balanced_accuracy

    for source in range(2):
        bits = (0,) * len(rosters[source])
        try:
            observed[source] = scores[source][0] = score(source, bits, (0,) * len(rosters[1 - source]))
        except classifiers.ClassificationUnavailable as error:
            issues.append(references.ReferenceIssue("observed_classifier", references.ANIMALS[source], bits, str(error)))
            return report(False)
    for index, assignment in enumerate(references.ASSIGNMENTS[1:], 1):
        classifiers._check(check)
        labels = assignment[:3], assignment[3:]
        for source in range(2):
            try:
                scores[source][index] = score(source, labels[source], labels[1 - source])
            except classifiers.ClassificationUnavailable as error:
                issues.append(references.ReferenceIssue("null_classifier", references.ANIMALS[source], labels[source], str(error)))
                return report(False)
    return report(True)
