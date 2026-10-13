"""Prospectively selected EP02 binomial/logistic-normal observation recipe.

One Gaussian mouse intercept is shared by every session of that mouse.
Mechanism logits are fixed caller inputs, with no second fitted readout
intercept. Fit the population offset by bounded marginal likelihood at each
caller-frozen SD-grid point, using only the eight training mice. A separate
final-refit entrypoint uses all nine fixed development controls after the
caller's legitimate recipe lock and reader release. It creates neither.
New-mouse
prediction integrates its intercept jointly across sessions; fitted population
hyperparameters remain plug-in estimates, not Bayesian posterior draws.

No source adapter, trial-to-session mapping, mechanism fitting, empirical
access, threshold selection or end-to-end model qualification is supplied.
Array identities cannot prove provenance or enforce the access firewall.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, roots_hermitenorm

from development_score_core import DEVELOPMENT_RECORDS, SEED_COUNT, DevelopmentScore


class ObservationError(ValueError):
    """Invalid recipe, incomplete fold or unresolved numerical convergence."""


@dataclass(frozen=True)
class IntegrationSettings:
    orders: tuple[int, ...]
    log_probability_tolerance: float
    score_tolerance: float
    offset_tolerance: float
    maximum_root_iterations: int
    maximum_matrix_cells: int


@dataclass(frozen=True)
class ObservationRecipe:
    offset_bounds: tuple[float, float]
    mouse_sd_grid: tuple[float, ...]
    profile_tie_tolerance: float
    integration: IntegrationSettings


@dataclass(frozen=True)
class TrainingMouse:
    mouse_id: int
    session_ids: tuple[str, ...]
    mechanism_logits: tuple[float, ...]
    successes: tuple[int, ...]
    trials: tuple[int, ...]


@dataclass(frozen=True)
class ProfilePoint:
    mouse_sd: float
    population_offset: float
    training_log_probability: float
    offset_on_boundary: bool
    quadrature_log_probability_disagreement: float
    quadrature_score_disagreement: float


@dataclass(frozen=True)
class ObservationFit:
    held_mouse_id: int
    training_mouse_ids: tuple[int, ...]
    recipe: ObservationRecipe
    profile: tuple[ProfilePoint, ...]
    selected_index: int

    @property
    def selected(self):
        return self.profile[self.selected_index]


@dataclass(frozen=True)
class FinalObservationFit:
    """All-nine development refit, not a LOSO fit or access/recipe-lock grant.

    Population parameters are constrained plug-in estimates. This artifact
    contains no posterior draws, propagated mechanism uncertainty, audit
    outcomes or permission to construct/open any intervention evaluation.
    """

    training_mouse_ids: tuple[int, ...]
    recipe: ObservationRecipe
    profile: tuple[ProfilePoint, ...]
    selected_index: int

    @property
    def selected(self):
        return self.profile[self.selected_index]


@dataclass(frozen=True)
class PopulationPrediction:
    """Outcome-free distribution specification; hyperparameters never refit."""

    held_mouse_id: int
    training_mouse_ids: tuple[int, ...]
    session_ids: tuple[str, ...]
    mechanism_logits: tuple[float, ...]
    population_offset: float
    mouse_sd: float
    recipe: ObservationRecipe

    @property
    def integration(self):
        return self.recipe.integration

    def log_probability(self, *, successes, trials):
        """Score outcomes only: one integral of the conditional session product.

        Every call uses the same predeclared quadrature orders, nodes and
        hyperparameters. Outcomes never change a distribution, integration
        order, parameter, or model. Failed accuracy checks refuse the score.
        """
        row = _mouse(TrainingMouse(self.held_mouse_id, self.session_ids,
                                  self.mechanism_logits, successes, trials))
        recipe = _recipe(self.recipe)
        settings = recipe.integration
        offset = _finite(self.population_offset, "population offset")
        sd = _finite(self.mouse_sd, "mouse SD", minimum=0)
        if sd not in recipe.mouse_sd_grid or not recipe.offset_bounds[0] <= offset <= recipe.offset_bounds[1]:
            raise ObservationError("prediction parameters differ from the supplied constrained recipe")
        return _integrate(row, offset, sd, settings, _rules(settings))[0]


@dataclass(frozen=True)
class PopulationMouseFold:
    held_mouse_id: int
    session_ids: tuple[str, ...]
    successes: tuple[int, ...]
    trials: tuple[int, ...]
    seed_ids: tuple[int, ...]
    candidate_predictions: tuple[PopulationPrediction, ...]
    constant_null_predictions: tuple[PopulationPrediction, ...]


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ObservationError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name, minimum=None, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ObservationError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum) or (positive and value <= 0):
        raise ObservationError(f"invalid {name}")
    return value


def _settings(value):
    if not isinstance(value, IntegrationSettings):
        raise ObservationError("explicit IntegrationSettings required")
    orders = tuple(_integer(q, "quadrature order", 2) for q in value.orders)
    if len(orders) < 2 or orders != tuple(sorted(set(orders))) or orders[-1] > 2048:
        raise ObservationError("at least two increasing quadrature orders <= 2048 required")
    return IntegrationSettings(
        orders, _finite(value.log_probability_tolerance, "log probability tolerance", positive=True),
        _finite(value.score_tolerance, "score tolerance", positive=True),
        _finite(value.offset_tolerance, "offset tolerance", positive=True),
        _integer(value.maximum_root_iterations, "root iteration limit", 1),
        _integer(value.maximum_matrix_cells, "matrix cell limit", 1))


def _recipe(value):
    if not isinstance(value, ObservationRecipe) or len(value.offset_bounds) != 2:
        raise ObservationError("explicit ObservationRecipe with two offset bounds required")
    bounds = tuple(_finite(v, "offset bound") for v in value.offset_bounds)
    grid = tuple(_finite(v, "mouse SD", minimum=0) for v in value.mouse_sd_grid)
    if bounds[0] >= bounds[1] or not grid or grid[0] != 0 or grid != tuple(sorted(set(grid))):
        raise ObservationError("ordered offset bounds and increasing SD grid starting at zero required")
    return ObservationRecipe(bounds, grid, _finite(value.profile_tie_tolerance,
                             "profile tie tolerance", minimum=0), _settings(value.integration))


def _mouse(value):
    if not isinstance(value, TrainingMouse):
        raise ObservationError("TrainingMouse required")
    mouse = _integer(value.mouse_id, "mouse ID", 1)
    names = tuple(value.session_ids)
    if not names or any(not isinstance(s, str) or not s for s in names) or len(set(names)) != len(names):
        raise ObservationError("nonempty unique explicit session IDs required")
    offsets = tuple(_finite(x, "fixed mechanism logit") for x in value.mechanism_logits)
    k = tuple(_integer(x, "successes") for x in value.successes)
    n = tuple(_integer(x, "trials") for x in value.trials)
    if len(offsets) != len(names) or len(k) != len(names) or len(n) != len(names):
        raise ObservationError("all session arrays must have the same length")
    if any(ki > ni or ni > 2**53 for ki, ni in zip(k, n)):
        raise ObservationError("invalid or inexactly representable session counts")
    if sum(n) == 0:
        raise ObservationError("each mouse requires a positive total number of trials")
    return TrainingMouse(mouse, names, offsets, k, n)


def _rules(settings):
    result = []
    for order in settings.orders:
        nodes, weights = roots_hermitenorm(order)
        # Underflowed tail weights carry no representable mass. Retain the
        # native quadrature normalization, without empirical reweighting.
        keep = weights > 0
        log_weights = np.log(weights[keep]) - 0.5 * math.log(2 * math.pi)
        result.append((nodes[keep], log_weights))
    return result


def _integrate(row, offset, sd, settings, rules):
    count = len(row.session_ids)
    if count * (1 if sd == 0 else settings.orders[-1]) > settings.maximum_matrix_cells:
        raise ObservationError("declared quadrature matrix resource limit exceeded")
    k, n = np.asarray(row.successes, dtype=float), np.asarray(row.trials, dtype=float)
    constants = math.fsum(0.0 if ki in (0, ni) else
                          math.lgamma(ni + 1) - math.lgamma(ki + 1) - math.lgamma(ni - ki + 1)
                          for ki, ni in zip(row.successes, row.trials))
    active_rules = [(np.array([0.0]), np.array([0.0]))] if sd == 0 else rules
    values = []
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for nodes, log_weights in active_rules:
                eta = np.asarray(row.mechanism_logits)[None, :] + offset + sd * nodes[:, None]
                conditional = constants - np.sum(k * np.logaddexp(0, -eta)
                                                 + (n - k) * np.logaddexp(0, eta), axis=1)
                e = np.exp(-np.abs(eta))
                residual = np.where(eta >= 0, (k - n) + n * e / (1 + e),
                                    k - n * e / (1 + e))
                log_mass = log_weights + conditional
                logp = float(logsumexp(log_mass))
                derivative = float(np.sum(np.exp(log_mass - logp) * np.sum(residual, axis=1)))
                if not math.isfinite(logp) or not math.isfinite(derivative):
                    raise ObservationError("nonfinite integrated likelihood or derivative")
                values.append((logp, derivative))
    except (FloatingPointError, OverflowError) as error:
        raise ObservationError("integrated likelihood is not numerically representable") from error
    # The zero-variance case is the exact original binomial product.
    error_p = 0.0 if sd == 0 else abs(values[-1][0] - values[-2][0])
    error_g = 0.0 if sd == 0 else abs(values[-1][1] - values[-2][1])
    if error_p > settings.log_probability_tolerance or error_g > settings.score_tolerance:
        raise ObservationError("predeclared quadrature orders did not converge")
    return values[-1][0], values[-1][1], error_p, error_g


def fit_observation_model(*, training_mice, held_mouse_id, recipe):
    """Constrained marginal MLE; all grid values must successfully qualify.

    Candidate/common offsets and SD grid are frozen caller inputs. Maximize
    over the bounded offset for each SD, then choose the smallest SD within
    the supplied numerical tie tolerance of the highest training likelihood.
    No per-mouse grid selection or held-mouse outcome is accepted.
    """
    recipe = _recipe(recipe)
    held = _integer(held_mouse_id, "held mouse ID", 1)
    if held not in DEVELOPMENT_RECORDS:
        raise ObservationError("held mouse must be a fixed development record")
    expected = tuple(m for m in DEVELOPMENT_RECORDS if m != held)
    rows = tuple(_mouse(m) for m in training_mice)
    if len(rows) != 8 or {m.mouse_id for m in rows} != set(expected):
        raise ObservationError("exactly the eight other development mice are required")
    rows = tuple(sorted(rows, key=lambda m: m.mouse_id))
    profile, chosen = _fit_profile(rows, recipe)
    return ObservationFit(held, expected, recipe, profile, chosen)


def refit_final_observation_model(*, training_mice, recipe, check):
    """Refit the locked observation recipe to exactly the nine control mice.

    The caller must already have a legitimate recipe lock and development
    reader release. Supplied fixed mechanism logits and verified session
    counts retain their existing roles; this array-only API cannot establish
    provenance, choose endpoint semantics, fit mechanisms or grant access.
    There is no held-mouse argument and no intervention outcome is accepted.
    The required budget callback may abort any evaluation; no partial result
    is returned. Final fitted hyperparameters remain conditional plug-in
    estimates, with the same finite SD grid and offset-bound limitations.
    """
    if not callable(check):
        raise ObservationError("an explicit final-refit budget callback is required")
    check()
    recipe = _recipe(recipe)
    rows = []
    for row in training_mice:
        check()
        if (len(rows) >= len(DEVELOPMENT_RECORDS) or not isinstance(row, TrainingMouse)
                or _integer(row.mouse_id, "mouse ID", 1) not in DEVELOPMENT_RECORDS):
            raise ObservationError("final refit requires exactly the nine fixed development mice")
        rows.append(_mouse(row))
    if len(rows) != len(DEVELOPMENT_RECORDS) or {row.mouse_id for row in rows} != set(DEVELOPMENT_RECORDS):
        raise ObservationError("final refit requires exactly the nine fixed development mice")
    rows = tuple(sorted(rows, key=lambda row: row.mouse_id))
    profile, chosen = _fit_profile(rows, recipe, check=check)
    check()
    return FinalObservationFit(DEVELOPMENT_RECORDS, recipe, profile, chosen)


def _fit_profile(rows, recipe, check=None):
    """The unchanged constrained profile algorithm, shared by fold and final fits."""
    budget_error = None

    def checkpoint():
        nonlocal budget_error
        if check is not None:
            try:
                check()
            except BaseException as error:
                # A callback's RuntimeError/ValueError must not be mistaken
                # for the numerical root failures translated below.
                budget_error = error
                raise

    checkpoint()
    rules, profile = _rules(recipe.integration), []
    checkpoint()
    lo, hi = recipe.offset_bounds
    for sd in recipe.mouse_sd_grid:
        checkpoint()
        def evaluate(offset):
            values = []
            for row in rows:
                checkpoint()
                values.append(_integrate(row, offset, sd, recipe.integration, rules))
            return (math.fsum(v[0] for v in values), math.fsum(v[1] for v in values),
                    max(v[2] for v in values), max(v[3] for v in values))

        left, right = evaluate(lo), evaluate(hi)
        if left[1] <= 0:
            optimum, boundary = lo, True
        elif right[1] >= 0:
            optimum, boundary = hi, True
        else:
            try:
                optimum, diagnostic = brentq(lambda b: evaluate(b)[1], lo, hi,
                    xtol=recipe.integration.offset_tolerance, rtol=4 * np.finfo(float).eps,
                    maxiter=recipe.integration.maximum_root_iterations, full_output=True, disp=False)
            except (ValueError, RuntimeError) as error:
                if error is budget_error:
                    raise
                raise ObservationError("bounded population-offset solve failed") from error
            if not diagnostic.converged:
                raise ObservationError("population-offset iteration limit reached")
            boundary = False
        result = evaluate(optimum)
        if not boundary and abs(result[1]) > 8 * recipe.integration.score_tolerance:
            raise ObservationError("population-offset score tolerance not met")
        if result[0] + 8 * recipe.integration.log_probability_tolerance < max(left[0], right[0]):
            raise ObservationError("interior fit is worse than a boundary")
        profile.append(ProfilePoint(sd, float(optimum), result[0], boundary, result[2], result[3]))
    best = max(point.training_log_probability for point in profile)
    chosen = next(i for i, point in enumerate(profile)
                  if best - point.training_log_probability <= recipe.profile_tie_tolerance)
    checkpoint()
    return tuple(profile), chosen


def predict_new_mouse(fit, *, held_mouse_id, session_ids, mechanism_logits):
    """Construct the held distribution without accepting any held outcomes."""
    if not isinstance(fit, ObservationFit) or held_mouse_id != fit.held_mouse_id:
        raise ObservationError("prediction requires the matching fitted development fold")
    names, logits = tuple(session_ids), tuple(mechanism_logits)
    # Reuse the structural validator with disposable counts, not observations.
    row = _mouse(TrainingMouse(held_mouse_id, names, logits, (0,) * len(names), (1,) * len(names)))
    selected = fit.selected
    return PopulationPrediction(fit.held_mouse_id, fit.training_mouse_ids, row.session_ids,
                                row.mechanism_logits, selected.population_offset,
                                selected.mouse_sd, fit.recipe)


def score_population_loso_panel(folds, *, frozen_complexity, configuration_id):
    """Integrate within mouse/seed, average five seeds, then weight mice equally."""
    complexity = _finite(frozen_complexity, "frozen complexity", minimum=0)
    if not isinstance(configuration_id, str) or not configuration_id:
        raise ObservationError("nonempty configuration ID required")
    folds = tuple(folds)
    if len(folds) != 9 or any(not isinstance(f, PopulationMouseFold) for f in folds):
        raise ObservationError("exactly nine PopulationMouseFold records required")
    by_mouse = {_integer(f.held_mouse_id, "held mouse ID", 1): f for f in folds}
    if len(by_mouse) != 9 or set(by_mouse) != set(DEVELOPMENT_RECORDS):
        raise ObservationError("fixed nine-mouse panel required")
    shared_seeds, common_recipe, gains = None, None, []
    for mouse in DEVELOPMENT_RECORDS:
        fold = by_mouse[mouse]
        row = _mouse(TrainingMouse(mouse, fold.session_ids, (0,) * len(fold.session_ids),
                                  fold.successes, fold.trials))
        seeds = tuple(_integer(s, "seed ID") for s in fold.seed_ids)
        if len(seeds) != SEED_COUNT or len(set(seeds)) != SEED_COUNT:
            raise ObservationError("five distinct shared seed IDs required")
        ordered = tuple(sorted(seeds))
        if shared_seeds is None:
            shared_seeds = ordered
        elif ordered != shared_seeds:
            raise ObservationError("seed bank must be identical across mice")
        scores = []
        for predictions in (fold.candidate_predictions, fold.constant_null_predictions):
            if len(predictions) != SEED_COUNT:
                raise ObservationError("five predictions per model required")
            model_scores = []
            for index in sorted(range(SEED_COUNT), key=seeds.__getitem__):
                prediction = predictions[index]
                if (not isinstance(prediction, PopulationPrediction)
                        or prediction.held_mouse_id != mouse or prediction.session_ids != row.session_ids
                        or prediction.training_mouse_ids != tuple(m for m in DEVELOPMENT_RECORDS if m != mouse)):
                    raise ObservationError("prediction fold, training roster or session roster differs")
                if common_recipe is None:
                    common_recipe = prediction.recipe
                elif prediction.recipe != common_recipe:
                    raise ObservationError("all models/folds/seeds require the same observation recipe")
                model_scores.append(prediction.log_probability(successes=row.successes, trials=row.trials)
                                    / sum(row.trials))
            scores.append(math.fsum(model_scores) / SEED_COUNT)
        gains.append(scores[0] - scores[1])
    return DevelopmentScore(min(gains), sorted(gains)[4], math.fsum(gains) / 9,
                            complexity, configuration_id)
