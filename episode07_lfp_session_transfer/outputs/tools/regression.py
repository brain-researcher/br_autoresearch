"""Explicit weighted multioutput mapping candidates for EP07.

Inputs must already be admitted training arrays. This numerical primitive does
not authenticate roles, mix native neuron spaces, choose hyperparameters, open
data, or score development/held-out outcomes. It launches no process or job.

Weights are nonnegative and normalized to sum one; loss sums over outputs:
  .5 * sum_i w_i ||Y_i - prediction_i||^2.
Ridge/RRR add .5 * penalty * ||B||_F^2. Elementwise elastic net adds
penalty*l1_ratio*||B||_1 + .5*penalty*(1-l1_ratio)*||B||_F^2.
fit_intercept=True centers X and Y by their weighted training means; the
linear intercept is mean_Y - mean_X @ B and is never penalized. There is no
implicit standardization. Zero-weight rows remain required finite inputs.

RRR restricts rank(B), using the right singular subspace of the augmented
ridge-fitted response, including penalty rows. Its rank is a regression
parameter, distinct from the population bridge's coordinate rank. Elastic net
uses deterministic cyclic coordinate descent and absolute coefficient KKT
tolerance. A freshly formed training residual verifies KKT before success and
at the iteration cap; Gram cancellation cannot by itself certify or reject a
solution. Exhausting the cap above tolerance raises, returning no model.

The shallow network is two affine layers with one tanh/relu hidden layer of
width <=64. It uses CPU float64, local-generator Xavier-uniform initialization,
full-batch Adam (betas .9/.999, epsilon 1e-8), and exactly the supplied epochs.
L2 penalizes the two kernels only. When intercept fitting is enabled, inputs
and outputs are weighted-centered and both layer biases are trainable; when
disabled, centers and biases are zero. Fixed epochs do not imply convergence.
Torch is imported only for network fitting. Prediction uses frozen NumPy
parameters. The caller owns BLAS/Torch thread limits and allocation budgets;
this module never mutates global thread configuration or random generators.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


@dataclass(frozen=True)
class RegressionRecipe:
    family: str
    penalty: float
    fit_intercept: bool = True
    l1_ratio: float | None = None
    rank: int | None = None
    max_iterations: int | None = None
    tolerance: float | None = None
    width: int | None = None
    activation: str | None = None
    seed: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None


@dataclass(frozen=True)
class NetworkParameters:
    hidden_kernel: np.ndarray
    hidden_bias: np.ndarray
    output_kernel: np.ndarray
    output_bias: np.ndarray


@dataclass(frozen=True)
class FittedRegression:
    recipe: RegressionRecipe
    input_mean: np.ndarray
    output_mean: np.ndarray
    coefficients: np.ndarray | None
    network: NetworkParameters | None
    training_rows: int
    positive_weight_rows: int
    iterations: int
    converged: bool | None
    stopping_reason: str
    objective: float


class RegressionNotConverged(RuntimeError):
    def __init__(self, iterations, kkt_error, *, gram_kkt_error, tolerance):
        self.iterations = iterations
        self.kkt_error = kkt_error
        self.gram_kkt_error = gram_kkt_error
        self.tolerance = tolerance
        super().__init__("elastic net exhausted its iteration limit before the specified KKT tolerance: "
                         f"iterations={iterations}, residual_kkt_error={kkt_error:.17g}, "
                         f"gram_kkt_error={gram_kkt_error:.17g}, tolerance={tolerance:.17g}")


def _check(check):
    if not callable(check):
        raise TypeError("a caller-owned cancellation/resource check is required")
    check()


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _real(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"invalid explicit {name}")


def _recipe(recipe):
    optional = {"l1_ratio", "rank", "max_iterations", "tolerance", "width",
                "activation", "seed", "learning_rate", "epochs"}
    allowed = {"ridge": set(), "elastic_net": {"l1_ratio", "max_iterations", "tolerance"},
               "reduced_rank_ridge": {"rank"},
               "shallow_network": {"width", "activation", "seed", "learning_rate", "epochs"}}
    if not isinstance(recipe, RegressionRecipe) or recipe.family not in allowed:
        raise ValueError("unknown regression recipe")
    _real(recipe.penalty, "penalty")
    if not isinstance(recipe.fit_intercept, bool):
        raise ValueError("fit_intercept must be an explicit boolean")
    if any(getattr(recipe, name) is not None for name in optional - allowed[recipe.family]):
        raise ValueError("recipe supplies parameters unused by its chosen family")
    if recipe.family == "elastic_net":
        _real(recipe.l1_ratio, "l1_ratio")
        if recipe.l1_ratio > 1:
            raise ValueError("l1_ratio must be in [0,1]")
        _integer(recipe.max_iterations, "max_iterations")
        _real(recipe.tolerance, "tolerance", positive=True)
    elif recipe.family == "reduced_rank_ridge":
        _integer(recipe.rank, "rank")
    elif recipe.family == "shallow_network":
        _integer(recipe.width, "width", maximum=64)
        _integer(recipe.seed, "seed", minimum=0, maximum=2**63 - 1)
        _integer(recipe.epochs, "epochs")
        _real(recipe.learning_rate, "learning_rate", positive=True)
        if recipe.activation not in ("tanh", "relu"):
            raise ValueError("network activation must be explicitly tanh or relu")
    return recipe


def _matrix(value, name):
    original = np.asarray(value)
    if original.ndim != 2 or min(original.shape) == 0 or original.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real two-dimensional array")
    array = np.array(original, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite in float64 arithmetic")
    return array


def _finite(value):
    if not np.isfinite(value).all():
        raise FloatingPointError("nonfinite regression arithmetic; no observations or features were dropped")
    return value


def _weights(value, rows):
    original = np.asarray(value)
    if (original.shape != (rows,) or original.dtype.kind not in "fiu"
            or not np.isfinite(original).all() or np.any(original < 0)):
        raise ValueError("one finite nonnegative weight is required per training row")
    weights = np.array(original, dtype=np.float64, copy=True)
    maximum = float(np.max(weights))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("training row weights must have positive finite mass")
    weights /= maximum
    weights /= np.sum(weights)
    return weights


def _ridge(x, y, weights, penalty, rank, check):
    _check(check)
    root = np.sqrt(weights)[:, None]
    a, b = x * root, y * root
    if penalty > 0:
        a = np.vstack((a, math.sqrt(penalty) * np.eye(x.shape[1])))
        b = np.vstack((b, np.zeros((x.shape[1], y.shape[1]))))
    _finite(a)
    _finite(b)
    _check(check)
    coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
    _finite(coefficients)
    if rank is not None:
        _check(check)
        fitted = _finite(a @ coefficients)
        _, _, vt = np.linalg.svd(fitted, full_matrices=False)
        projection = vt[:rank].T @ vt[:rank]
        coefficients = _finite(coefficients @ projection)
    _check(check)
    return coefficients


def _kkt_error(gradient, coefficients, l1):
    violations = np.where(coefficients != 0,
        np.abs(gradient + l1 * np.sign(coefficients)), np.maximum(np.abs(gradient) - l1, 0))
    return float(np.max(_finite(violations)))


def _residual_kkt_error(x, y, weights, coefficients, l1, l2, check):
    """Coefficient-space KKT from a fresh residual, in the original fit units.

    This is the same objective as the Gram updates, with no rescaling or
    relative tolerance. Re-forming the residual avoids cancellation of the
    two separately formed Gram/cross products in the convergence decision.
    It is a float64 numerical check, not an exact-arithmetic certificate.
    """
    _check(check)
    residual = _finite(x @ coefficients - y)
    gradient = _finite(x.T @ (weights[:, None] * residual) + l2 * coefficients)
    _check(check)
    return _kkt_error(gradient, coefficients, l1)


def _elastic_net(x, y, weights, recipe, check):
    _check(check)
    gram = _finite(x.T @ (weights[:, None] * x))
    cross = _finite(x.T @ (weights[:, None] * y))
    coefficients = np.zeros((x.shape[1], y.shape[1]), dtype=np.float64)
    l1 = float(recipe.penalty * recipe.l1_ratio)
    l2 = float(recipe.penalty * (1 - recipe.l1_ratio))
    for iteration in range(1, recipe.max_iterations + 1):
        for column in range(x.shape[1]):
            _check(check)
            denominator = float(_finite(gram[column, column] + l2))
            if denominator == 0:
                coefficients[column] = 0
                continue
            rho = cross[column] - gram[column] @ coefficients + gram[column, column] * coefficients[column]
            coefficients[column] = np.sign(rho) * np.maximum(np.abs(rho) - l1, 0) / denominator
            _finite(coefficients[column])
        _check(check)
        gradient = _finite(gram @ coefficients - cross + l2 * coefficients)
        gram_error = _kkt_error(gradient, coefficients, l1)
        if gram_error <= recipe.tolerance or iteration == recipe.max_iterations:
            error = _residual_kkt_error(x, y, weights, coefficients, l1, l2, check)
            if error <= recipe.tolerance:
                return coefficients, iteration
    raise RegressionNotConverged(recipe.max_iterations, error,
        gram_kkt_error=gram_error, tolerance=float(recipe.tolerance))


def _network_fit(x, y, weights, recipe, check):
    _check(check)
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(recipe.seed))
    tx = torch.tensor(x, dtype=torch.float64, device="cpu")
    ty = torch.tensor(y, dtype=torch.float64, device="cpu")
    tw = torch.tensor(weights[:, None], dtype=torch.float64, device="cpu")
    def kernel(rows, columns):
        limit = math.sqrt(6 / (rows + columns))
        values = (2 * torch.rand((rows, columns), dtype=torch.float64, device="cpu",
                                generator=generator) - 1) * limit
        return torch.nn.Parameter(values)
    hidden = kernel(x.shape[1], int(recipe.width))
    output = kernel(int(recipe.width), y.shape[1])
    hidden_bias = torch.zeros(int(recipe.width), dtype=torch.float64, device="cpu")
    output_bias = torch.zeros(y.shape[1], dtype=torch.float64, device="cpu")
    parameters = [hidden, output]
    if recipe.fit_intercept:
        hidden_bias, output_bias = torch.nn.Parameter(hidden_bias), torch.nn.Parameter(output_bias)
        parameters.extend((hidden_bias, output_bias))
    optimizer = torch.optim.Adam(parameters, lr=float(recipe.learning_rate), betas=(.9, .999),
                                 eps=1e-8, weight_decay=0, amsgrad=False, foreach=False)
    for _ in range(recipe.epochs):
        _check(check)
        optimizer.zero_grad(set_to_none=True)
        activation = tx @ hidden + hidden_bias
        activation = torch.tanh(activation) if recipe.activation == "tanh" else torch.relu(activation)
        residual = activation @ output + output_bias - ty
        loss = .5 * torch.sum(tw * residual.square())
        loss = loss + .5 * recipe.penalty * (hidden.square().sum() + output.square().sum())
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("nonfinite network training objective")
        loss.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item())
               for parameter in parameters):
            raise FloatingPointError("nonfinite network gradient")
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all().item()) for parameter in parameters):
            raise FloatingPointError("nonfinite network parameter")
    _check(check)
    return NetworkParameters(_readonly(hidden.detach().numpy()), _readonly(hidden_bias.detach().numpy()),
                             _readonly(output.detach().numpy()), _readonly(output_bias.detach().numpy()))


def _network_predict(parameters, x, activation):
    hidden = x @ parameters.hidden_kernel + parameters.hidden_bias
    _finite(hidden)
    hidden = np.tanh(hidden) if activation == "tanh" else np.maximum(hidden, 0)
    return _finite(hidden @ parameters.output_kernel + parameters.output_bias)


def fit_regression(X, Y, *, row_weights, recipe, check):
    """Fit one explicit candidate; arrays alone do not establish training roles."""
    _check(check)
    recipe = _recipe(recipe)
    x, y = _matrix(X, "X"), _matrix(Y, "Y")
    if len(x) != len(y):
        raise ValueError("X and Y training rows must correspond exactly")
    if recipe.family == "reduced_rank_ridge" and recipe.rank > min(x.shape[1], y.shape[1]):
        raise ValueError("regression rank exceeds the input/output dimensions")
    weights = _weights(row_weights, len(x))
    _check(check)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        input_mean = np.sum(weights[:, None] * x, axis=0) if recipe.fit_intercept else np.zeros(x.shape[1])
        output_mean = np.sum(weights[:, None] * y, axis=0) if recipe.fit_intercept else np.zeros(y.shape[1])
        x, y = _finite(x - input_mean), _finite(y - output_mean)
        coefficients = network = None
        if recipe.family in ("ridge", "reduced_rank_ridge"):
            coefficients = _ridge(x, y, weights, recipe.penalty, recipe.rank, check)
            iterations, converged, reason = 1, True, "closed_form"
        elif recipe.family == "elastic_net":
            coefficients, iterations = _elastic_net(x, y, weights, recipe, check)
            converged, reason = True, "absolute_coefficient_kkt_tolerance"
        else:
            network = _network_fit(x, y, weights, recipe, check)
            iterations, converged, reason = recipe.epochs, None, "fixed_epochs_no_convergence_claim"
        prediction = x @ coefficients if coefficients is not None else _network_predict(network, x, recipe.activation)
        residual = _finite(prediction - y)
        objective = .5 * np.sum(weights[:, None] * residual**2)
        if network is not None:
            objective += .5 * recipe.penalty * (np.sum(network.hidden_kernel**2) + np.sum(network.output_kernel**2))
        elif recipe.family == "elastic_net":
            objective += recipe.penalty * recipe.l1_ratio * np.sum(np.abs(coefficients))
            objective += .5 * recipe.penalty * (1 - recipe.l1_ratio) * np.sum(coefficients**2)
        else:
            objective += .5 * recipe.penalty * np.sum(coefficients**2)
        _finite(objective)
    _check(check)
    return FittedRegression(recipe, _readonly(input_mean), _readonly(output_mean),
        None if coefficients is None else _readonly(coefficients), network,
        len(x), int(np.count_nonzero(weights)), int(iterations), converged, reason, float(objective))


def predict_regression(fitted, X, *, check):
    """Apply frozen coefficients; no role permission is inferred from X."""
    _check(check)
    if not isinstance(fitted, FittedRegression):
        raise TypeError("expected one fitted regression candidate")
    _recipe(fitted.recipe)
    x = _matrix(X, "prediction X")
    if fitted.input_mean.shape != (x.shape[1],) or fitted.output_mean.ndim != 1:
        raise ValueError("prediction feature dimension differs from the fitted mapping")
    with np.errstate(over="ignore", invalid="ignore"):
        x = _finite(x - fitted.input_mean)
        if fitted.recipe.family == "shallow_network":
            if fitted.network is None or fitted.coefficients is not None:
                raise ValueError("fitted network parameters are incomplete")
            h, q = fitted.recipe.width, len(fitted.output_mean)
            for array, shape in ((fitted.network.hidden_kernel, (x.shape[1], h)),
                                 (fitted.network.hidden_bias, (h,)),
                                 (fitted.network.output_kernel, (h, q)),
                                 (fitted.network.output_bias, (q,))):
                if array.shape != shape:
                    raise ValueError("fitted network axes are inconsistent")
                _finite(array)
            values = _network_predict(fitted.network, x, fitted.recipe.activation)
        else:
            if fitted.network is not None or fitted.coefficients is None:
                raise ValueError("fitted linear coefficients are incomplete")
            if fitted.coefficients.shape != (x.shape[1], len(fitted.output_mean)):
                raise ValueError("fitted coefficient axes are inconsistent")
            values = x @ fitted.coefficients
        values = _finite(values + fitted.output_mean)
    _check(check)
    return _readonly(values)
