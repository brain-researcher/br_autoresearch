"""EP06 source-animal classifiers for complete paired regional profiles.

This array-only primitive does not read neural data, choose sessions, search,
run a permutation test, authorize an opening, or certify scientific support.
The caller supplies its complete prospective session roster and explicit
per-session binary labels. Ordinary labels are M1=1 and PMd=0; a paired null
must supply its swapped labels explicitly to fitting AND scoring.

Each regional profile is centered over its declared ordered feature blocks
and divided by its L2 norm. There is no across-session or target-trained
transformation. Prototype fitting averages normalized rows equally within
each source class, then normalizes the two means for cosine comparison.
Logistic fitting minimizes mean row logloss + .5*lambda*||beta||^2, with an
unpenalized intercept. Newton/Armijo uses at most 100 iterations and 40 step
halvings per iteration, and requires max(abs(gradient)) <= 1e-9.

An exact decision-score tie has no predicted label and receives half credit
in balanced accuracy. Zero/nonfinite shapes or prototypes and numerical fit
failure make the entire supplied comparison unavailable: no row, session or
feature is dropped. Unavailable evidence must not be recoded as a score of 0.
Physical feature blocks must match across all sessions; regional population
axis identities remain independent. Actual training provenance and episode
data-access/minimum-support rules remain the caller's responsibility.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from bandwise_prediction import BandwiseRecipe, FeatureBlock, _recipe
from component_scoring import FrozenDesign
from primary_profile import ProfileScore, SignedProfile


class ClassificationUnavailable(RuntimeError):
    """Complete declared comparison cannot be estimated; never drop its rows."""


@dataclass(frozen=True)
class PairedSessionProfiles:
    animal: str
    profile: SignedProfile


@dataclass(frozen=True)
class PairedLabels:
    animal: str
    session_id: str
    m1_label: int
    pmd_label: int


@dataclass(frozen=True)
class ClassifierRecipe:
    method: str                    # cosine_prototype or l2_logistic
    penalty: float | None = None   # explicit positive lambda for logistic only


@dataclass(frozen=True)
class FittedClassifier:
    source_animal: str
    source_sessions: tuple[str, ...]
    source_labels: tuple[PairedLabels, ...]
    blocks: tuple[FeatureBlock, ...]
    recipe: ClassifierRecipe
    prototypes: np.ndarray | None  # label 0 then label 1
    coefficients: np.ndarray | None
    intercept: float | None
    objective: float | None
    iterations: int
    maximum_absolute_gradient: float | None


@dataclass(frozen=True)
class PredictionRow:
    session_id: str
    region: str
    margin: float                  # positive means class 1; zero is an exact tie
    predicted_label: int | None


@dataclass(frozen=True)
class ProfilePredictions:
    source_animal: str
    source_sessions: tuple[str, ...]
    target_animal: str
    target_sessions: tuple[str, ...]
    blocks: tuple[FeatureBlock, ...]
    rows: tuple[PredictionRow, ...]  # explicit session order, M1 then PMd


@dataclass(frozen=True)
class ClassificationScore:
    balanced_accuracy: float
    class_0_recall: float
    class_1_recall: float
    tied_rows: int
    total_rows: int


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _roster(animal, expected_sessions):
    if not isinstance(animal, str) or not animal:
        raise ValueError("explicit animal identity required")
    if (not isinstance(expected_sessions, tuple) or not expected_sessions
            or any(not isinstance(session, str) or not session for session in expected_sessions)
            or len(set(expected_sessions)) != len(expected_sessions)):
        raise ValueError("complete, ordered, nonempty and unique session roster required")
    return expected_sessions


def _labels(labels, animal, sessions):
    if not isinstance(labels, tuple) or len(labels) != len(sessions):
        raise ValueError("one explicit paired-label record per declared session required")
    values = []
    for session, label in zip(sessions, labels):
        if (not isinstance(label, PairedLabels) or label.animal != animal
                or label.session_id != session
                or any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                       for value in (label.m1_label, label.pmd_label))
                or {label.m1_label, label.pmd_label} != {0, 1}):
            raise ValueError("labels must preserve the explicit animal/session order and paired {0,1} support")
        values.extend((int(label.m1_label), int(label.pmd_label)))
    return np.asarray(values, dtype=np.float64)


def _classifier_recipe(recipe):
    if not isinstance(recipe, ClassifierRecipe):
        raise TypeError("explicit classifier recipe required")
    if recipe.method == "cosine_prototype" and recipe.penalty is None:
        return recipe
    if (recipe.method == "l2_logistic" and not isinstance(recipe.penalty, (bool, np.bool_))
            and isinstance(recipe.penalty, Real) and math.isfinite(float(recipe.penalty))
            and recipe.penalty > 0):
        return ClassifierRecipe(recipe.method, float(recipe.penalty))
    raise ValueError("cosine prototype has no penalty; L2 logistic requires an explicit positive finite penalty")


def _profile_metadata(profiles, animal, sessions, expected_blocks=None):
    if not isinstance(profiles, tuple) or len(profiles) != len(sessions):
        raise ValueError("complete paired profiles required for the caller's exact session roster")
    blocks = expected_blocks
    for session, supplied in zip(sessions, profiles):
        if (not isinstance(supplied, PairedSessionProfiles) or supplied.animal != animal
                or not isinstance(supplied.profile, SignedProfile)):
            raise ValueError("explicit animal-bound paired profiles required")
        pair = supplied.profile
        if pair.session_id != session:
            raise ValueError("paired profile order must match the caller's complete session roster")
        left, right = pair.m1, pair.pmd
        if (not isinstance(left, ProfileScore) or not isinstance(right, ProfileScore)
                or left.region != "M1" or right.region != "PMd"):
            raise ValueError("every session requires M1 and PMd in their physical region roles")
        a, b = left.design, right.design
        if (not isinstance(a, FrozenDesign) or not isinstance(b, FrozenDesign)
                or a.role != "evaluation" or b.role != "evaluation"):
            raise PermissionError("primary classifier profiles must have evaluation whole-trial roles")
        if (a.session_id != session or b.session_id != session
                or any(getattr(a, name) != getattr(b, name) for name in
                       ("directions", "trial_ids", "trial_directions", "time_ids"))
                or len(a.axis_ids) != len(b.axis_ids)):
            raise ValueError("paired session profiles need identical trial/direction/time support and equal rank")
        current = _recipe(BandwiseRecipe(1, 1., pair.blocks)).blocks
        if (current != tuple(value.block for value in left.blocks)
                or current != tuple(value.block for value in right.blocks)
                or (blocks is not None and current != blocks)):
            raise ValueError("all regional and session rows require identical ordered physical feature blocks")
        blocks = current
    return blocks


def _shape_matrix(profiles, check):
    rows = []
    for supplied in profiles:
        for profile in (supplied.profile.m1, supplied.profile.pmd):
            _check(check)
            values = tuple(block.score for block in profile.blocks)
            if any(value is None for value in values):
                raise ClassificationUnavailable("required_profile_block_unavailable")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
                raise ValueError("profile scores must be explicit real numbers or unavailable")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                vector = np.asarray(values, dtype=np.float64)
                vector = vector - np.mean(vector)
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
                    raise ClassificationUnavailable("zero_or_nonfinite_centered_profile_shape")
                rows.append(vector / norm)
    return np.asarray(rows, dtype=np.float64)


def _logistic_state(parameters, x, y, penalty):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        beta, intercept = parameters[:-1], parameters[-1]
        logit = x @ beta + intercept
        objective = float(np.mean(np.logaddexp(0., logit) - y * logit)
                          + .5 * penalty * np.dot(beta, beta))
        exponential = np.exp(-np.abs(logit))
        probability = np.where(logit >= 0, 1. / (1. + exponential), exponential / (1. + exponential))
        augmented = np.column_stack((x, np.ones(len(y))))
        gradient = augmented.T @ (probability - y) / len(y)
        gradient[:-1] += penalty * beta
        weight = probability * (1. - probability)
        hessian = augmented.T @ (weight[:, None] * augmented) / len(y)
        hessian[:-1, :-1] += penalty * np.eye(x.shape[1])
    if (not math.isfinite(objective) or not np.isfinite(gradient).all()
            or not np.isfinite(hessian).all()):
        raise ClassificationUnavailable("nonfinite_logistic_arithmetic")
    return objective, gradient, hessian


def _fit_logistic(x, y, penalty, check):
    parameters = np.zeros(x.shape[1] + 1, dtype=np.float64)
    for iteration in range(101):
        _check(check)
        objective, gradient, hessian = _logistic_state(parameters, x, y, penalty)
        maximum_gradient = float(np.max(np.abs(gradient)))
        if maximum_gradient <= 1e-9:
            return parameters, objective, iteration, maximum_gradient
        if iteration == 100:
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise ClassificationUnavailable("singular_logistic_newton_system") from exc
        decrement = float(np.dot(gradient, step))
        if not np.isfinite(step).all() or not math.isfinite(decrement) or decrement <= 0:
            raise ClassificationUnavailable("invalid_logistic_newton_direction")
        for halving in range(41):
            _check(check)
            rate = math.ldexp(1., -halving)
            candidate = parameters - rate * step
            candidate_objective, _, _ = _logistic_state(candidate, x, y, penalty)
            if candidate_objective <= objective - 1e-4 * rate * decrement:
                parameters = candidate
                break
        else:
            raise ClassificationUnavailable("logistic_line_search_exhausted")
    raise ClassificationUnavailable("logistic_iteration_limit_without_required_convergence")


def fit_profile_classifier(profiles, *, source_animal, expected_sessions, labels, recipe, check):
    """Fit one caller-specified source animal; every session supplies both regions."""
    _check(check)
    sessions = _roster(source_animal, expected_sessions)
    recipe = _classifier_recipe(recipe)
    y = _labels(labels, source_animal, sessions)
    blocks = _profile_metadata(profiles, source_animal, sessions)
    x = _shape_matrix(profiles, check)
    if recipe.method == "cosine_prototype":
        prototypes = np.stack((np.mean(x[y == 0], axis=0), np.mean(x[y == 1], axis=0)))
        norms = np.linalg.norm(prototypes, axis=1)
        if not np.isfinite(prototypes).all() or not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ClassificationUnavailable("zero_or_nonfinite_source_class_prototype")
        _check(check)
        return FittedClassifier(source_animal, sessions, labels, blocks, recipe,
                                _readonly(prototypes / norms[:, None]), None, None, None, 0, None)
    parameters, objective, iterations, gradient = _fit_logistic(x, y, recipe.penalty, check)
    _check(check)
    return FittedClassifier(source_animal, sessions, labels, blocks, recipe, None,
                            _readonly(parameters[:-1]), float(parameters[-1]), objective, iterations, gradient)


def predict_profile_classifier(fitted, profiles, *, target_animal, expected_sessions, check):
    """Apply a frozen classifier; no target label or target-trained fit is used."""
    _check(check)
    if not isinstance(fitted, FittedClassifier):
        raise TypeError("frozen source-animal classifier required")
    _roster(fitted.source_animal, fitted.source_sessions)
    _labels(fitted.source_labels, fitted.source_animal, fitted.source_sessions)
    recipe = _classifier_recipe(fitted.recipe)
    sessions = _roster(target_animal, expected_sessions)
    if target_animal == fitted.source_animal or set(sessions) & set(fitted.source_sessions):
        raise ValueError("cross-animal prediction requires a different animal and disjoint session identities")
    blocks = _profile_metadata(profiles, target_animal, sessions, fitted.blocks)
    x = _shape_matrix(profiles, check)
    if recipe.method == "cosine_prototype":
        prototypes = np.asarray(fitted.prototypes)
        if (prototypes.dtype.kind != "f" or prototypes.shape != (2, len(blocks))
                or not np.isfinite(prototypes).all()
                or not np.allclose(np.linalg.norm(prototypes, axis=1), 1., rtol=0, atol=1e-12)):
            raise ValueError("frozen cosine classifier requires two finite unit-norm source prototypes")
        margins = x @ prototypes[1] - x @ prototypes[0]
    else:
        coefficients = np.asarray(fitted.coefficients)
        if (coefficients.dtype.kind != "f" or coefficients.shape != (len(blocks),)
                or not np.isfinite(coefficients).all() or fitted.intercept is None
                or not math.isfinite(fitted.intercept)
                or fitted.maximum_absolute_gradient is None
                or not math.isfinite(fitted.maximum_absolute_gradient)
                or fitted.maximum_absolute_gradient > 1e-9):
            raise ValueError("frozen logistic classifier must retain its finite converged source fit")
        margins = x @ coefficients + fitted.intercept
    if not np.isfinite(margins).all():
        raise ClassificationUnavailable("nonfinite_classifier_decision_score")
    rows = tuple(PredictionRow(session, region, float(margin),
                              None if margin == 0 else int(margin > 0))
                 for (session, region), margin in zip(
                     ((session, region) for session in sessions for region in ("M1", "PMd")), margins))
    _check(check)
    return ProfilePredictions(fitted.source_animal, fitted.source_sessions,
                              target_animal, sessions, blocks, rows)


def score_profile_predictions(predictions, *, labels, check):
    """Balanced accuracy under explicit paired target labels, with half-credit ties."""
    _check(check)
    if not isinstance(predictions, ProfilePredictions):
        raise TypeError("complete source-bound cross-animal predictions required")
    sessions = _roster(predictions.target_animal, predictions.target_sessions)
    _roster(predictions.source_animal, predictions.source_sessions)
    if (predictions.source_animal == predictions.target_animal
            or set(sessions) & set(predictions.source_sessions)):
        raise ValueError("cross-animal prediction identities must remain distinct")
    y = _labels(labels, predictions.target_animal, sessions)
    expected = tuple((session, region) for session in sessions for region in ("M1", "PMd"))
    if (not isinstance(predictions.rows, tuple) or len(predictions.rows) != len(expected)
            or any(not isinstance(row, PredictionRow) or (row.session_id, row.region) != identity
                   for row, identity in zip(predictions.rows, expected))):
        raise ValueError("both ordered regional predictions required for every declared target session")
    correct = []
    for row, truth in zip(predictions.rows, y):
        _check(check)
        if (not isinstance(row.margin, Real) or not math.isfinite(row.margin)
                or row.predicted_label != (None if row.margin == 0 else int(row.margin > 0))):
            raise ValueError("finite margins and exact-tie prediction labels must agree")
        correct.append(.5 if row.predicted_label is None else float(row.predicted_label == truth))
    correct = np.asarray(correct)
    class_0, class_1 = float(np.mean(correct[y == 0])), float(np.mean(correct[y == 1]))
    return ClassificationScore(.5 * (class_0 + class_1), class_0, class_1,
                               sum(row.predicted_label is None for row in predictions.rows), len(correct))
