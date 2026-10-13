"""Array-only quadratic-MAP binomial optimizer, with fixed development folds.

This is a numerical component, not an adopted observation-model recipe. The
caller supplies the entire design, session offsets, prior center and strictly
positive-definite penalty. No intercept, mouse effect, prior strength, trial to
session mapping, or data reader is supplied here. Mouse labels check the
declared fold; they cannot authenticate array provenance or enforce isolation.

The returned coefficient estimate and held-mouse logits are conditional plug-in
quantities. They do not integrate parameter or new-mouse-effect uncertainty and
therefore do not establish the policy's population-predictive distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from development_score_core import DEVELOPMENT_RECORDS


class FitError(ValueError):
    """Invalid inputs or failure to achieve the supplied numerical tolerance."""


@dataclass(frozen=True)
class NewtonSettings:
    max_iterations: int
    gradient_tolerance: float
    max_backtracks: int
    armijo_fraction: float
    step_contraction: float


@dataclass(frozen=True)
class BinomialMapFit:
    coefficients: tuple[float, ...]
    held_mouse_id: int
    training_mouse_ids: tuple[int, ...]
    session_count: int
    trial_count: int
    iterations: int
    negative_log_kernel_plus_penalty: float
    max_abs_gradient: float


def _integer(value, name, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise FitError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value, name, *, less_than=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0
            or (less_than is not None and value >= less_than)):
        raise FitError(f"invalid {name}")
    return float(value)


def _array(value, name, ndim):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "biuf":
            raise FitError(f"{name} must be a real numeric array")
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise FitError(f"invalid {name}") from error
    if result.ndim != ndim or not np.isfinite(result).all():
        raise FitError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _counts(values, name, minimum):
    result = tuple(_integer(value, name, minimum=minimum) for value in values)
    # Every supplied integer must be exactly representable by the float solver.
    if any(value > 2**53 for value in result):
        raise FitError(f"{name} exceeds exact float integer range")
    return np.array(result, dtype=float), sum(result)


def _evaluate(beta, design, offsets, successes, trials, precision, center):
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        eta = offsets + design @ beta
        delta = beta - center
        loss = (successes @ np.logaddexp(0.0, -eta)
                + (trials - successes) @ np.logaddexp(0.0, eta)
                + 0.5 * delta @ precision @ delta)
        exponential = np.exp(-np.abs(eta))
        # Evaluate the residual directly in each half-line, retaining the small
        # all-success tail even when sigmoid(eta) rounds to exactly one.
        residual = np.where(eta >= 0,
                            (trials - successes) - trials * exponential / (1 + exponential),
                            trials * exponential / (1 + exponential) - successes)
        curvature = trials * exponential / (1 + exponential)**2
        gradient = design.T @ residual + precision @ delta
        hessian = design.T @ (curvature[:, None] * design) + precision
    if (not np.isfinite(loss) or not np.isfinite(gradient).all()
            or not np.isfinite(hessian).all()):
        raise FloatingPointError("non-finite objective or derivatives")
    return float(loss), gradient, hessian


def fit_grouped_binomial_map(*, design, offsets, successes, trials,
                             row_mouse_ids, training_mouse_ids, held_mouse_id,
                             penalty_precision, prior_center, initial_coefficients,
                             settings: NewtonSettings) -> BinomialMapFit:
    """Minimize summed session binomial negative log kernel plus fixed quadratic.

    eta = offsets + design @ beta; penalty = (beta-center)' P (beta-center)/2.
    The omitted binomial coefficient is parameter-independent; use the existing
    development scorer for full log probabilities and per-trial normalization.
    Every one of the specified fold's eight training mice must have a row.
    No session is dropped, and no probability clipping or denominator scaling
    is performed. A successful return requires the supplied gradient tolerance.
    """
    held = _integer(held_mouse_id, "held mouse", minimum=1)
    if held not in DEVELOPMENT_RECORDS:
        raise FitError("held mouse must be a fixed development record")
    expected = tuple(mouse for mouse in DEVELOPMENT_RECORDS if mouse != held)
    declared = tuple(_integer(mouse, "training mouse", minimum=1) for mouse in training_mouse_ids)
    rows = tuple(_integer(mouse, "row mouse", minimum=1) for mouse in row_mouse_ids)
    if len(declared) != 8 or set(declared) != set(expected) or set(rows) != set(expected):
        raise FitError("fit requires exactly the eight training mice, with no held-mouse rows")
    x, offset = _array(design, "design", 2), _array(offsets, "offsets", 1)
    center = _array(prior_center, "prior center", 1)
    beta = _array(initial_coefficients, "initial coefficients", 1)
    precision = _array(penalty_precision, "penalty precision", 2)
    k, _ = _counts(successes, "successes", 0)
    n, total_trials = _counts(trials, "trials", 1)
    size, dimensions = x.shape
    if (dimensions < 1 or size != len(rows) or offset.shape != (size,)
            or k.shape != (size,) or n.shape != (size,)
            or center.shape != (dimensions,) or beta.shape != (dimensions,)
            or precision.shape != (dimensions, dimensions) or np.any(k > n)):
        raise FitError("incompatible arrays or successes exceeding trials")
    if not np.array_equal(precision, precision.T):
        raise FitError("penalty precision must be symmetric; no implicit repair")
    try:
        np.linalg.cholesky(precision)
    except np.linalg.LinAlgError as error:
        raise FitError("penalty precision must be strictly positive definite") from error
    if not isinstance(settings, NewtonSettings):
        raise FitError("explicit NewtonSettings required")
    max_iterations = _integer(settings.max_iterations, "max iterations", minimum=1)
    max_backtracks = _integer(settings.max_backtracks, "max backtracks", minimum=1)
    tolerance = _positive(settings.gradient_tolerance, "gradient tolerance")
    armijo = _positive(settings.armijo_fraction, "Armijo fraction", less_than=1)
    contraction = _positive(settings.step_contraction, "step contraction", less_than=1)

    try:
        for iteration in range(max_iterations + 1):
            loss, gradient, hessian = _evaluate(beta, x, offset, k, n, precision, center)
            norm = float(np.max(np.abs(gradient)))
            if norm <= tolerance:
                return BinomialMapFit(tuple(float(value) for value in beta), held, expected,
                                      size, total_trials, iteration, loss, norm)
            if iteration == max_iterations:
                break
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                direction = np.linalg.solve(hessian, -gradient)
                slope = float(gradient @ direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                raise FitError("Newton direction is not finite descent")
            step = 1.0
            for _ in range(max_backtracks):
                try:
                    with np.errstate(over="raise", invalid="raise", under="ignore"):
                        candidate = beta + step * direction
                    candidate_loss, _, _ = _evaluate(candidate, x, offset, k, n, precision, center)
                except FloatingPointError:
                    candidate_loss = np.inf
                if candidate_loss <= loss + armijo * step * slope:
                    beta = candidate
                    break
                step *= contraction
            else:
                raise FitError("bounded Newton line search failed")
    except (FloatingPointError, np.linalg.LinAlgError) as error:
        raise FitError("objective or Newton solve is not numerically representable") from error
    raise FitError("maximum iterations reached without the supplied gradient tolerance")


def held_mouse_plugin_logits(fit: BinomialMapFit, *, held_mouse_id, design, offsets):
    """Apply fitted coefficients to caller-fixed rows; never accepts outcomes.

    The caller must fix population/new-mouse feature rows without target outcome
    fitting. This generic function cannot certify those semantics or turn MAP
    logits into an integrated population-predictive distribution.
    """
    held = _integer(held_mouse_id, "held mouse", minimum=1)
    if held != fit.held_mouse_id:
        raise FitError("prediction mouse differs from the specified held fold")
    x, offset = _array(design, "prediction design", 2), _array(offsets, "prediction offsets", 1)
    if not len(x) or x.shape[1] != len(fit.coefficients) or offset.shape != (len(x),):
        raise FitError("incompatible prediction arrays")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            logits = offset + x @ np.asarray(fit.coefficients)
    except FloatingPointError as error:
        raise FitError("prediction is not numerically representable") from error
    if not np.isfinite(logits).all():
        raise FitError("non-finite prediction")
    return tuple(float(value) for value in logits)
