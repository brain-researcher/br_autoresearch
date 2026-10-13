"""EP19 classical references and logistic hazard, with no source I/O.

Implements the *existing* section-5 recipes. Callers still own source admission,
causal preprocessing, onset/risk labels, data isolation and the pre-score freeze.
No sampling rate, FIR, sensor unit, calibration, margin or source role is guessed.
Inputs are already scaled by the common fitting-only voltage transform; features
are centered/scaled again using this model's fitting examples only.

The two-summary head is older then recent. Tangent-reference fitting pools the
two slot covariances with equal slot weight within each decision's person weight.
Original bandpower filters the uninterrupted past; the factorial adapter resets
its FIR at the slot boundary and discards its declared FIR warm-up, never pads.
A slot lacking enough supported samples is unresolved, not silently shortened.
The predictor retains no cross-example state: each caller supplies the complete
past window (including FIR history where required) from one reset boundary.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit, logsumexp

from forecast_components import censoring_loss_and_logit_gradient
from model_recipes import ReferenceRecipe, reference_recipes, self_paced_folds, way_inner_folds


class UnresolvedFit(RuntimeError):
    """A required fit/representation failed; it may not be dropped or replaced."""


def _matrix(value, columns=None):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("a nonempty finite matrix is required")
    if columns is not None and result.shape[1] != columns:
        raise ValueError(f"expected {columns} columns")
    return result


def _weights(weights, rows):
    weights = np.asarray(weights, dtype=np.float64)
    if (weights.shape != (rows,) or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)):
        raise ValueError("every retained fitting point needs a finite positive weight")
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("fitting weight total must be finite and positive")
    return weights / total


def _labels(category, complete, rows):
    category, complete = np.asarray(category), np.asarray(complete)
    if (category.shape != (rows,) or complete.shape != (rows,)
            or not np.issubdtype(category.dtype, np.integer)
            or not np.issubdtype(complete.dtype, np.integer)
            or np.any((category < -1) | (category > 18))
            or np.any((complete < 0) | (complete > 18))):
        raise ValueError("integer categories -1..18 and complete bins 0..18 required")
    return category.astype(np.int64), complete.astype(np.int64)


def participant_weights(participants):
    """Equal total per person, equal weight per natural at-risk point within it."""
    people = tuple(participants)
    if not people or any(person is None for person in people):
        raise ValueError("each fitting point must have its declared participant")
    counts = {person: people.count(person) for person in set(people)}
    return np.asarray([1. / (len(counts) * counts[person]) for person in people])


@dataclass(frozen=True)
class GroupIdentity:
    source: str
    participant: Any
    series_or_run: Any
    boundary_id: str


@dataclass(frozen=True)
class FittingScope:
    source: str
    fold_id: str
    allowed_participants: tuple
    allowed_series: tuple | None

    @classmethod
    def development(cls, registry, *, source, fold_id, participant=None):
        if source == "WAY":
            folds = way_inner_folds(registry, participant)
            matching = [fold for fold in folds if fold.fold_id == fold_id]
            if not matching:
                raise ValueError("unknown whole-series fitting fold")
            return cls(source, fold_id, (participant,), matching[0].fit)
        if source == "self_paced":
            folds = self_paced_folds(registry)
            matching = [fold for fold in folds if fold.fold_id == fold_id]
            if not matching:
                raise ValueError("unknown whole-person fitting fold")
            return cls(source, fold_id, matching[0].fit, None)
        raise ValueError("only the two declared development sources are supported")

    @classmethod
    def final_refit(cls, registry, *, source, participant=None):
        """Registered final fitting cohort; never a held-out evaluation role.

        Consumers must explicitly opt into this scope. Existing development-only
        validators continue to reject it unless their interface is extended.
        """
        definition = registry["folds"]
        if source == "WAY":
            way_inner_folds(registry, participant)
            series = tuple(definition["WAY_final_refit_series"])
            if (series != (1, 2, 3, 4, 5, 6, 7)
                    or tuple(definition["WAY_feedback_series"]) != (7,)
                    or set(series) & set(definition["WAY_held_out_series"])):
                raise ValueError("WAY final fitting requires exactly series1–7, disjoint held-out series")
            return cls(source, f"WAY.P{participant}.final_refit_series1_7", (participant,), series)
        if source == "self_paced":
            if participant is not None:
                raise ValueError("self-paced final fitting is one complete group fit")
            folds = self_paced_folds(registry)
            people = tuple(sorted(person for fold in folds for person in fold.validation))
            return cls(source, "self_paced.final_refit_all15", people, None)
        raise ValueError("only the two declared final fitting sources are supported")

    def validate(self, identities):
        identities = tuple(identities)
        if not identities:
            raise ValueError("empty fitting fold")
        for identity in identities:
            if (identity.source != self.source
                    or identity.participant not in self.allowed_participants
                    or not identity.boundary_id
                    or (self.allowed_series is not None
                        and identity.series_or_run not in self.allowed_series)):
                raise ValueError("fitting example crosses a whole-group role boundary")
        # Do not turn a missing/failed person or series into a successful smaller fold.
        if set(item.participant for item in identities) != set(self.allowed_participants):
            raise UnresolvedFit("required fitting participant is absent")
        if self.allowed_series is not None and set(item.series_or_run for item in identities) != set(self.allowed_series):
            raise UnresolvedFit("required fitting series is absent")
        # Boundary identifiers are globally unique in this fold, never reused for
        # another person's series, even when timestamps happen to look continuous.
        bindings = {}
        for item in identities:
            binding = (item.source, item.participant, item.series_or_run)
            if item.boundary_id in bindings and bindings[item.boundary_id] != binding:
                raise ValueError("one boundary identifier names different recording groups")
            bindings[item.boundary_id] = binding
        return participant_weights([item.participant for item in identities])


def canonical_reference_scope(registry, scope, *, fitting_stage="development"):
    """Bind an EEG fitting stage without opening a data role.

    Final refitting is an explicit opt-in to the already registered cohort.
    Default development consumers continue to reject every final-refit scope.
    This helper is here so EEG consumers need no reverse baseline import.
    """
    if not isinstance(scope, FittingScope):
        raise ValueError("explicit canonical EEG fitting scope required")
    if fitting_stage not in ("development", "final_refit"):
        raise ValueError("explicit development or final_refit fitting stage required")
    if scope.source == "WAY" and len(scope.allowed_participants) != 1:
        raise ValueError("canonical fitting scope requires one WAY participant")
    participant = scope.allowed_participants[0] if scope.source == "WAY" else None
    if fitting_stage == "final_refit":
        expected = FittingScope.final_refit(registry, source=scope.source, participant=participant)
    else:
        expected = FittingScope.development(registry, source=scope.source,
            fold_id=scope.fold_id, participant=participant)
    if scope != expected:
        role = "development fold" if fitting_stage == "development" else "final_refit cohort"
        raise ValueError("supplied fitting scope differs from its canonical " + role)
    return expected


@dataclass(frozen=True)
class PastWindow:
    values: np.ndarray
    sample_times: np.ndarray
    boundary_ids: Sequence[str]

    def validated(self, *, decision_time, sample_rate_hz, history_seconds,
                  boundary_id, extra_history_samples=0):
        values = _matrix(self.values, 29)
        times = np.asarray(self.sample_times, dtype=np.float64)
        count = exact_samples(history_seconds, sample_rate_hz)
        expected = count + extra_history_samples
        if len(values) != expected or times.shape != (expected,):
            raise ValueError("complete fixed history, including explicit FIR warm-up, required")
        if not np.isfinite(decision_time) or not np.all(np.isfinite(times)):
            raise ValueError("finite timestamps required")
        if (len(self.boundary_ids) != expected or not boundary_id
                or any(value != boundary_id for value in self.boundary_ids)):
            raise ValueError("window crosses an acquisition gap, series or role boundary")
        # No future sample, duplicate, implicit interpolation or end padding. The
        # scientific grid here is half-open, exactly [t-history, t).
        wanted = decision_time - np.arange(expected, 0, -1) / sample_rate_hz
        tolerance = max(1e-10, abs(decision_time) * np.finfo(float).eps * 8)
        if np.any(times >= decision_time) or not np.allclose(times, wanted, atol=tolerance, rtol=0):
            raise ValueError("window is not the exact contiguous past-only analysis grid")
        return values


def exact_samples(seconds, sample_rate_hz):
    if not np.isfinite(seconds) or not np.isfinite(sample_rate_hz) or seconds <= 0 or sample_rate_hz <= 0:
        raise ValueError("explicit positive duration and admitted sampling rate required")
    value = seconds * sample_rate_hz
    rounded = int(round(value))
    if rounded < 1 or not np.isclose(value, rounded, rtol=0, atol=1e-9):
        raise ValueError("declared duration must be exactly representable on the analysis grid")
    return rounded


def normalized_log_probabilities(logits):
    logits = _matrix(logits, 19)
    return logits - logsumexp(logits, axis=1, keepdims=True)


def ensemble_log_probabilities(seed_log_probabilities):
    """Average all three registered seed probabilities, never pick a seed."""
    values = np.asarray(seed_log_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != 19 or not np.all(np.isfinite(values)):
        raise ValueError("all three finite 19-category seed results are required")
    if not np.allclose(logsumexp(values, axis=2), 0, atol=1e-12, rtol=0):
        raise ValueError("seed probability vectors must be normalized")
    result = logsumexp(values, axis=0) - np.log(3.)
    return normalized_log_probabilities(result)


def hazard_log_probabilities(logits):
    logits = _matrix(logits, 18)
    log_hazard = -np.logaddexp(0., -logits)
    log_survival = -np.logaddexp(0., logits)
    prior = np.c_[np.zeros(len(logits)), np.cumsum(log_survival, axis=1)]
    return np.c_[log_hazard + prior[:, :18], prior[:, 18]]


def hazard_loss_and_gradient(logits, category, complete):
    """Same censored 19-category likelihood in conditional-hazard coordinates."""
    logits = _matrix(logits, 18)
    category, complete = _labels(category, complete, len(logits))
    bins = np.arange(18)[None, :]
    observed = np.where(category < 0, complete, np.minimum(category + 1, 18))
    at_risk = bins < observed[:, None]
    event = (bins == category[:, None]) & (category[:, None] >= 0)
    loss = np.sum(at_risk * (np.logaddexp(0., logits) - event * logits), axis=1)
    gradient = at_risk * (expit(logits) - event)
    return loss, gradient


@dataclass
class LinearFit:
    coefficients: np.ndarray
    intercept: np.ndarray
    converged: bool
    iterations: int
    objective: float
    initial_objective: float
    reason: str

    def require_converged(self):
        if not self.converged:
            raise UnresolvedFit(self.reason)
        return self


def _proximal_fit(features, category, complete, weights, *, penalty, l1_ratio,
                  offsets=None, hazard=False, maximum_iterations=2000,
                  relative_tolerance=1e-6):
    """Declared proximal gradient/backtracking, no validation or alternate fit."""
    x = _matrix(features)
    category, complete = _labels(category, complete, len(x))
    weights = _weights(weights, len(x))
    if (not np.isfinite(penalty) or penalty < 0 or not np.isfinite(l1_ratio)
            or not 0 <= l1_ratio <= 1 or maximum_iterations < 1
            or not np.isfinite(relative_tolerance) or relative_tolerance <= 0):
        raise ValueError("invalid registered optimizer parameters")
    columns = 18 if hazard else 19
    if hazard:
        if offsets is not None or l1_ratio != 0:
            raise ValueError("registered logistic hazard has no EEG offset or L1 term")
        offset = np.zeros((len(x), columns))
    else:
        offset = _matrix(offsets, 19)
        if len(offset) != len(x) or not np.allclose(logsumexp(offset, axis=1), 0, atol=1e-12, rtol=0):
            raise ValueError("frozen fitting-fold baseline log probabilities required")
    w, b = np.zeros((x.shape[1], columns)), np.zeros(columns)
    ridge, lasso = penalty * (1 - l1_ratio), penalty * l1_ratio

    def smooth(coefficients, intercept, gradient=False):
        logits = x @ coefficients + intercept + offset
        if hazard:
            loss, grad = hazard_loss_and_gradient(logits, category, complete)
        else:
            loss, grad = censoring_loss_and_logit_gradient(
                normalized_log_probabilities(logits), category, complete)
        objective = float(weights @ loss + .5 * ridge * np.sum(coefficients ** 2))
        if not gradient:
            return objective
        grad = grad * weights[:, None]
        return objective, x.T @ grad + ridge * coefficients, grad.sum(axis=0)

    objective = smooth(w, b) + lasso * np.abs(w).sum()
    initial = float(objective)
    step = 1.
    for iteration in range(1, maximum_iterations + 1):
        base, grad_w, grad_b = smooth(w, b, gradient=True)
        accepted = False
        for _ in range(60):
            candidate_w = w - step * grad_w
            candidate_w = np.sign(candidate_w) * np.maximum(np.abs(candidate_w) - step * lasso, 0)
            candidate_b = b - step * grad_b
            delta_w, delta_b = candidate_w - w, candidate_b - b
            candidate_smooth = smooth(candidate_w, candidate_b)
            quadratic = base + np.sum(grad_w * delta_w) + grad_b @ delta_b + (
                np.sum(delta_w ** 2) + delta_b @ delta_b) / (2 * step)
            if np.isfinite(candidate_smooth) and candidate_smooth <= quadratic + 1e-12:
                accepted = True
                break
            step *= .5
        if not accepted:
            return LinearFit(w, b, False, iteration, float(objective), initial, "backtracking_failed")
        updated = float(candidate_smooth + lasso * np.abs(candidate_w).sum())
        if not np.isfinite(updated) or updated > objective + 1e-10:
            return LinearFit(w, b, False, iteration, float(objective), initial, "nonfinite_or_nonmonotone_objective")
        relative = abs(objective - updated) / max(1., abs(objective))
        w, b, objective = candidate_w, candidate_b, updated
        if relative <= relative_tolerance:
            return LinearFit(w, b, True, iteration, objective, initial, "relative_objective_tolerance")
        step = min(step * 1.1, 1.)
    return LinearFit(w, b, False, maximum_iterations, float(objective), initial, "iteration_limit")


def fit_elastic_correction(features, category, complete, weights, *, offsets,
                           penalty, l1_ratio):
    return _proximal_fit(features, category, complete, weights, offsets=offsets,
                         penalty=penalty, l1_ratio=l1_ratio)


def fit_logistic_hazard(features, category, complete, weights, *, penalty):
    if penalty not in (.001, .01, .1):
        raise ValueError("logistic-hazard penalty must be one of the three registered values")
    return _proximal_fit(features, category, complete, weights, penalty=penalty,
                         l1_ratio=0., hazard=True)


class FittingStandardizer:
    def fit(self, features, weights):
        features = _matrix(features)
        weights = _weights(weights, len(features))
        self.mean = weights @ features
        self.scale = np.sqrt(weights @ ((features - self.mean) ** 2))
        # An exactly constant fitting feature carries no scale information. Unit
        # divisor leaves it zero in fitting data without using validation values.
        constant = np.ptp(features, axis=0) == 0
        self.mean[constant] = features[0, constant]
        self.scale[constant | (self.scale == 0)] = 1.
        return self

    def transform(self, features):
        features = _matrix(features, len(self.mean))
        return (features - self.mean) / self.scale


def shrunk_covariance(voltage, shrinkage):
    voltage = _matrix(voltage, 29)
    if len(voltage) < 2 or shrinkage not in (.1, .5):
        raise ValueError("two supported samples and a registered shrinkage required")
    centered = voltage - voltage.mean(axis=0)
    covariance = centered.T @ centered / len(voltage)
    return ((1 - shrinkage) * covariance
            + (shrinkage * np.trace(covariance) / 29 + 1e-6) * np.eye(29))


def _symmetric(value):
    return (value + value.T) / 2


def _spd_function(value, function):
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(value))
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise UnresolvedFit("nonpositive covariance/reference; no eigenvalue repair")
    return _symmetric((eigenvectors * function(eigenvalues)) @ eigenvectors.T)


def _symmetric_exp(value):
    eigenvalues, eigenvectors = np.linalg.eigh(_symmetric(value))
    result = (eigenvectors * np.exp(eigenvalues)) @ eigenvectors.T
    if not np.all(np.isfinite(result)):
        raise UnresolvedFit("nonfinite affine-invariant mean update")
    return _symmetric(result)


def affine_invariant_mean(covariances, weights):
    covariances = np.asarray(covariances, dtype=np.float64)
    if (covariances.ndim != 3 or covariances.shape[1:] != (29, 29)
            or not np.all(np.isfinite(covariances))):
        raise ValueError("fitting-only 29-channel covariance matrices required")
    weights = _weights(weights, len(covariances))
    reference = np.einsum("n,nij->ij", weights, covariances)

    def coordinates(candidate):
        inverse = _spd_function(candidate, lambda values: 1 / np.sqrt(values))
        logs = np.asarray([_spd_function(inverse @ matrix @ inverse, np.log)
                           for matrix in covariances])
        objective = float(np.einsum("n,nij,nij->", weights, logs, logs))
        return np.einsum("n,nij->ij", weights, logs), objective

    for iteration in range(100):
        direction, objective = coordinates(reference)
        if np.linalg.norm(direction, "fro") <= 1e-8:
            return reference
        root = _spd_function(reference, np.sqrt)
        step = 1.
        for _ in range(40):
            updated = _symmetric(root @ _symmetric_exp(step * direction) @ root)
            _, next_objective = coordinates(updated)
            if next_objective <= objective + 1e-12:
                reference = updated
                break
            step *= .5
        else:
            raise UnresolvedFit("affine-invariant mean line search failed")
    direction, _ = coordinates(reference)
    if np.linalg.norm(direction, "fro") <= 1e-8:
        return reference
    raise UnresolvedFit("affine-invariant mean exceeded 100 iterations")


class TangentFeatures:
    def __init__(self, shrinkage):
        if shrinkage not in (.1, .5):
            raise ValueError("unregistered covariance shrinkage")
        self.shrinkage = shrinkage

    def fit(self, slots, weights):
        covariances = [shrunk_covariance(slot, self.shrinkage)
                       for pair in slots for slot in pair]
        self.reference = affine_invariant_mean(covariances, np.repeat(weights, 2) / 2)
        self.inverse_root = _spd_function(self.reference, lambda values: 1 / np.sqrt(values))
        return self

    def transform(self, slots):
        upper = np.triu_indices(29)
        multiplier = np.where(upper[0] == upper[1], 1., np.sqrt(2.))
        features = []
        for pair in slots:
            vectors = []
            for slot in pair:
                covariance = shrunk_covariance(slot, self.shrinkage)
                tangent = _spd_function(self.inverse_root @ covariance @ self.inverse_root, np.log)
                vectors.append(tangent[upper] * multiplier)
            features.append(np.concatenate(vectors))
        return _matrix(features, 870)


class BandpowerFeatures:
    """Five explicitly supplied causal FIRs; no coefficient design occurs here."""
    def __init__(self, coefficients):
        self.coefficients = tuple(np.asarray(item, dtype=np.float64) for item in coefficients)
        if len(self.coefficients) != 5 or any(item.ndim != 1 or not len(item)
                or not np.all(np.isfinite(item)) for item in self.coefficients):
            raise ValueError("five finite causal band FIRs in the registered band order required")
        self.warmup = max(len(item) for item in self.coefficients) - 1

    def _filtered(self, values, start, end, coefficients):
        if start < len(coefficients) - 1 or start >= end:
            raise ValueError("FIR slot lacks supported history; no zero/future padding")
        return sum(coefficient * values[start - lag:end - lag]
                   for lag, coefficient in enumerate(coefficients))

    def transform_original(self, histories, recent_samples):
        features = []
        for history in histories:
            cut = len(history) - recent_samples
            if cut <= self.warmup:
                raise ValueError("older slot lacks complete causal filter support")
            pieces = []
            for start, end in ((self.warmup, cut), (cut, len(history))):
                pieces.append(np.concatenate([np.log1p(np.mean(
                    self._filtered(history, start, end, coefficients) ** 2, axis=0))
                    for coefficients in self.coefficients]))
            features.append(np.concatenate(pieces))
        return _matrix(features, 290)

    def transform_factorial(self, slots):
        features = []
        for pair in slots:
            pieces = []
            for slot in pair:
                if len(slot) <= self.warmup:
                    raise ValueError("factorial slot cannot support its reset FIR warm-up")
                pieces.append(np.concatenate([np.log1p(np.mean(
                    self._filtered(slot, self.warmup, len(slot), coefficients) ** 2, axis=0))
                    for coefficients in self.coefficients]))
            features.append(np.concatenate(pieces))
        return _matrix(features, 290)


@dataclass(frozen=True)
class FittingExample:
    past_input: PastWindow
    decision_time: float
    identity: GroupIdentity
    observed_category: int
    complete_bins: int
    context_log_probabilities: np.ndarray


class ClassicalForecastModel:
    """Executable source-free implementation of two registered reference families.

    `semantics` must provide the registry, a canonical FittingScope, an admitted
    analysis rate and `adapter` ('original' or 'factorial'). Bandpower additionally
    requires the five causal FIRs. Factorial windows are *already reconstructed*
    fixed-shape masked slots, from forecast_components.factorized_blocks. This
    backend never receives the unmasked cache and never chooses a surrogate.
    The caller must fit a fresh object for every matched condition and seed.
    fitting_stage defaults to development; final_refit must be explicit.
    """
    def fit(self, batches, *, recipe: ReferenceRecipe, seed: int, semantics: Mapping[str, Any]):
        self.fitted = False
        scope = semantics["fitting_scope"]
        self.fitting_stage = semantics.get("fitting_stage", "development")
        scope = canonical_reference_scope(semantics["registry"], scope,
                                          fitting_stage=self.fitting_stage)
        registered = {item.recipe_id: item for item in reference_recipes(semantics["registry"])}
        if recipe.recipe_id not in registered or recipe != registered[recipe.recipe_id] or seed not in recipe.seeds:
            raise ValueError("recipe and seed must match the unchanged finite registry")
        self.recipe, self.seed = recipe, seed
        self.rate = semantics["analysis_sample_rate_hz"]
        self.history = exact_samples(recipe.history_seconds, self.rate)
        self.recent = exact_samples(.2, self.rate)
        self.adapter = semantics["adapter"]
        if self.adapter not in ("original", "factorial"):
            raise ValueError("explicit original/factorial adapter required")
        if recipe.family == "past_only_filterbank_bandpower_elastic_net":
            self.features = BandpowerFeatures(semantics["causal_filter_coefficients"])
            self.extra = self.features.warmup if self.adapter == "original" else 0
            l1_ratio = recipe.family_parameters["l1_ratio"]
        elif recipe.family == "past_only_covariance_riemannian_tangent_model":
            self.features = TangentFeatures(recipe.family_parameters["covariance_shrinkage"])
            self.extra, l1_ratio = 0, 0.
        else:
            raise ValueError("this backend implements only the two classical reference families")
        examples = tuple(example for batch in batches for example in batch)
        weights = scope.validate([example.identity for example in examples])
        histories = [self._window(example.past_input, example.decision_time, example.identity.boundary_id)
                     for example in examples]
        slots = self._slots(histories)
        if isinstance(self.features, TangentFeatures):
            self.features.fit(slots, weights)
        raw_features = self._features(histories)
        self.standardizer = FittingStandardizer().fit(raw_features, weights)
        self.fit_result = fit_elastic_correction(self.standardizer.transform(raw_features),
            [example.observed_category for example in examples],
            [example.complete_bins for example in examples], weights,
            offsets=[example.context_log_probabilities for example in examples],
            penalty=recipe.regularization, l1_ratio=l1_ratio)
        self.fitting_scope = scope
        self.fitted = self.fit_result.converged
        self.boundary_id = None
        return {"converged": self.fitted, "iterations": self.fit_result.iterations,
                "objective": self.fit_result.objective, "reason": self.fit_result.reason,
                "recipe_id": recipe.recipe_id, "seed": seed, "fold_id": scope.fold_id,
                "fitting_stage": self.fitting_stage,
                "examples": len(examples), "source_admission_established_by_backend": False}

    def _window(self, window, decision_time, boundary_id):
        return window.validated(decision_time=decision_time, sample_rate_hz=self.rate,
            history_seconds=self.recipe.history_seconds, boundary_id=boundary_id,
            extra_history_samples=self.extra)

    def _slots(self, histories):
        return [(history[self.extra:len(history) - self.recent], history[-self.recent:])
                for history in histories]

    def _features(self, histories):
        if isinstance(self.features, TangentFeatures):
            return self.features.transform(self._slots(histories))
        if self.adapter == "original":
            return self.features.transform_original(histories, self.recent)
        return self.features.transform_factorial(self._slots(histories))

    def reset(self, *, boundary_id):
        if not isinstance(boundary_id, str) or not boundary_id:
            raise ValueError("explicit nonempty reset boundary required")
        self.boundary_id = boundary_id
        self.last_decision_time = -np.inf

    def predict_log_probabilities(self, past_input, *, decision_time,
                                  context_log_probabilities):
        if not getattr(self, "fitted", False):
            raise UnresolvedFit("cannot predict from an incomplete or nonconverged fit")
        if self.boundary_id is None or decision_time <= self.last_decision_time:
            raise ValueError("reset boundary and strictly increasing decision times required")
        history = self._window(past_input, decision_time, self.boundary_id)
        features = self.standardizer.transform(self._features([history]))
        offset = _matrix(np.asarray(context_log_probabilities)[None, :], 19)
        if not np.allclose(logsumexp(offset, axis=1), 0, atol=1e-12, rtol=0):
            raise ValueError("normalized frozen baseline log probabilities required")
        logits = features @ self.fit_result.coefficients + self.fit_result.intercept + offset
        prediction = normalized_log_probabilities(logits)[0]
        self.last_decision_time = decision_time
        return prediction
