#!/usr/bin/env python3
"""Run the EP10 v3 M0 model with the frozen v2 numerical-solver addendum.

This module imports the untouched v1 engine and replaces only its deterministic
L-BFGS implementation.  Scientific inputs, design construction, likelihood,
ridge path, resampling, predictions, and output semantics remain in the v1
engine.  ``--probe`` is restricted to the one prospectively named development
training split and emits aggregate solver diagnostics only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
if (
    not SCRIPT_PATH.is_file()
    or SCRIPT_PATH.is_symlink()
    or SCRIPT_PATH != SCRIPT_PATH.resolve(strict=True)
):
    raise ImportError("v2 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
BASE_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0.py"
BASE_CONTRACT_PATH = SCRIPT_DIR.parent / "v3_m0_model_contract.yaml"
ADDENDUM_PATH = SCRIPT_DIR.parent / "v3_m0_solver_addendum_v2.yaml"

if (
    not BASE_ENGINE_PATH.is_file()
    or BASE_ENGINE_PATH.is_symlink()
    or BASE_ENGINE_PATH != BASE_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("frozen v1 engine must be an unsymlinked regular stage0 file")

_BASE_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_base_v1", BASE_ENGINE_PATH
)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"cannot import frozen v1 engine: {BASE_ENGINE_PATH}")
base = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base
_BASE_SPEC.loader.exec_module(base)


PARAMETER_DIMENSION = 255
HISTORY_SIZE = 255
MAXIMUM_ITERATIONS = 2000
GRADIENT_TOLERANCE = 1.0e-7
RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-10
ARMIJO_C1 = 1.0e-4
BACKTRACK_FACTOR = 0.5
LINE_SEARCH_MAXIMUM_EVALUATIONS = 80
INITIAL_STEP = 1.0
CURVATURE_FLOOR_MULTIPLIER = 1.0e-14
DESCENDING_LAMBDAS = tuple(sorted(base.LAMBDAS, reverse=True))
PROBE_OUTER_HELD_GROUP = "212074"
PROBE_INNER_HELD_GROUP = "220187"
PROBE_EXPECTED_CELLS = 153
PROBE_EXPECTED_GROUPS = 10
REPEAT_OBJECTIVE_TOLERANCE = 1.0e-12
REPEAT_COEFFICIENT_TOLERANCE = 1.0e-10
REPEAT_PREDICTION_TOLERANCE = 1.0e-10
ZERO_START_MARGINAL_TOLERANCE = 1.0e-5
ZERO_START_LOG_PROBABILITY_TOLERANCE = 1.0e-4
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = 1000


FIT_TELEMETRY: list[dict[str, Any]] = []


@dataclass(frozen=True)
class OptimizerResultV2:
    coefficients: np.ndarray
    objective: float
    gradient_infinity_norm: float
    iterations: int
    accepted_steps: int
    relative_objective_change: float
    solver_telemetry: Mapping[str, Any]


def _as_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise base.ModelError(f"v2 solver addendum has invalid {label}") from error
    if not math.isfinite(result):
        raise base.ModelError(f"v2 solver addendum has nonfinite {label}")
    return result


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the untouched base contract and the numerical-only addendum."""
    for observed, expected, label in (
        (base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"),
        (addendum_path, ADDENDUM_PATH, "v2 solver addendum"),
    ):
        absolute = Path(os.path.abspath(observed))
        expected_absolute = Path(os.path.abspath(expected))
        if (
            not absolute.is_file()
            or absolute.is_symlink()
            or absolute != absolute.resolve(strict=True)
            or expected_absolute != expected_absolute.resolve(strict=True)
            or absolute != expected_absolute
        ):
            raise base.ModelError(
                f"{label} must be the exact unsymlinked EP10 stage0 file"
            )

    base_contract = base.validate_contract(base_contract_path)
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v2"
        or addendum.get("status")
        != "frozen_before_v2_numerical_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v2 solver addendum identity or sealed scope changed")

    reference = addendum.get("base_contract", {})
    if (
        reference.get("required_basename") != base_contract_path.name
        or reference.get("required_schema_version")
        != base_contract.get("schema_version")
        or reference.get("required_status") != base_contract.get("status")
    ):
        raise base.ModelError("v2 addendum no longer references the frozen v1 contract")

    base_optimizer = base_contract.get("estimation", {}).get("optimizer", {})
    if (
        base_optimizer.get("algorithm") != "deterministic_L_BFGS"
        or base_optimizer.get("line_search") != "Armijo_backtracking"
        or _as_float(base_optimizer.get("armijo_c1"), "base Armijo c1")
        != ARMIJO_C1
        or _as_float(
            base_optimizer.get("backtrack_factor"), "base backtrack factor"
        )
        != BACKTRACK_FACTOR
        or _as_float(
            base_optimizer.get("gradient_infinity_tolerance"),
            "base gradient tolerance",
        )
        != GRADIENT_TOLERANCE
        or _as_float(
            base_optimizer.get("relative_objective_tolerance"),
            "base relative-objective tolerance",
        )
        != RELATIVE_OBJECTIVE_TOLERANCE
        or base_optimizer.get("convergence_requires_both_tolerances") is not True
        or base_optimizer.get("fallback_optimizer") != "forbidden"
    ):
        raise base.ModelError("an unchanged v1 optimizer field drifted")

    solver = addendum.get("solver", {})
    line_search = solver.get("line_search", {})
    convergence = solver.get("convergence", {})
    curvature = solver.get("curvature", {})
    if (
        solver.get("algorithm") != "deterministic_L_BFGS"
        or int(solver.get("parameter_dimension", -1)) != PARAMETER_DIMENSION
        or int(solver.get("history_size", -1)) != HISTORY_SIZE
        or int(solver.get("maximum_iterations", -1)) != MAXIMUM_ITERATIONS
        or line_search.get("algorithm") != "Armijo_backtracking"
        or _as_float(line_search.get("initial_step"), "initial step")
        != INITIAL_STEP
        or _as_float(line_search.get("armijo_c1"), "Armijo c1") != ARMIJO_C1
        or _as_float(line_search.get("backtrack_factor"), "backtrack factor")
        != BACKTRACK_FACTOR
        or int(line_search.get("maximum_evaluations", -1))
        != LINE_SEARCH_MAXIMUM_EVALUATIONS
        or _as_float(
            convergence.get("gradient_infinity_tolerance"),
            "gradient tolerance",
        )
        != GRADIENT_TOLERANCE
        or _as_float(
            convergence.get("relative_objective_tolerance"),
            "relative-objective tolerance",
        )
        != RELATIVE_OBJECTIVE_TOLERANCE
        or convergence.get("requires_both") is not True
        or _as_float(curvature.get("floor_multiplier"), "curvature floor")
        != CURVATURE_FLOOR_MULTIPLIER
        or curvature.get("skipped_pair_action") != "continue_without_pair"
        or solver.get("fallback_optimizer") != "forbidden"
        or solver.get("nonconvergence_or_line_search_failure") != "stop"
    ):
        raise base.ModelError("v2 solver settings changed")

    ridge_path = addendum.get("ridge_path", {})
    addendum_lambdas = tuple(
        float(value) for value in ridge_path.get("ordered_lambdas", ())
    )
    initialization = ridge_path.get("initialization", {})
    if (
        addendum_lambdas != DESCENDING_LAMBDAS
        or tuple(float(value) for value in base_contract["estimation"]["lambda_grid"])
        != base.LAMBDAS
        or initialization.get("first_lambda") != "all_zero_coefficients"
        or initialization.get("later_lambdas")
        != "immediately_preceding_converged_coefficients"
        or ridge_path.get("auxiliary_lambdas") != "forbidden"
    ):
        raise base.ModelError("v2 ridge grid or warm-start order changed")

    probe = addendum.get("targeted_probe", {})
    repeat = probe.get("repeat_checks", {})
    zero_start = probe.get("zero_start_checks", {})
    if (
        str(probe.get("outer_held_group")) != PROBE_OUTER_HELD_GROUP
        or str(probe.get("inner_held_group")) != PROBE_INNER_HELD_GROUP
        or int(probe.get("expected_training_cells", -1)) != PROBE_EXPECTED_CELLS
        or int(probe.get("expected_training_groups", -1))
        != PROBE_EXPECTED_GROUPS
        or probe.get("full_descending_path_required") is not True
        or int(probe.get("deterministic_warm_path_repeats", -1)) != 2
        or probe.get("zero_start_comparison_at_every_lambda") is not True
        or int(probe.get("warm_path_max_iterations_exclusive", -1))
        != WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
        or probe.get("held_out_scoring") != "forbidden"
        or probe.get("coefficient_or_prediction_vector_emission") != "forbidden"
        or probe.get("final_or_unassigned_access") != "forbidden"
        or _as_float(
            repeat.get("objective_absolute_tolerance"),
            "repeat objective tolerance",
        )
        != REPEAT_OBJECTIVE_TOLERANCE
        or _as_float(
            repeat.get("coefficient_max_absolute_tolerance"),
            "repeat coefficient tolerance",
        )
        != REPEAT_COEFFICIENT_TOLERANCE
        or _as_float(
            repeat.get("prediction_max_absolute_tolerance"),
            "repeat prediction tolerance",
        )
        != REPEAT_PREDICTION_TOLERANCE
        or _as_float(
            zero_start.get("marginal_max_absolute_tolerance"),
            "zero-start marginal tolerance",
        )
        != ZERO_START_MARGINAL_TOLERANCE
        or _as_float(
            zero_start.get("log_probability_max_absolute_tolerance"),
            "zero-start log-probability tolerance",
        )
        != ZERO_START_LOG_PROBABILITY_TOLERANCE
    ):
        raise base.ModelError("v2 targeted-probe contract changed")
    return base_contract, addendum


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
            "last": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "last": float(array[-1]),
    }


def _solver_telemetry(
    *,
    status: str,
    ridge_lambda: float,
    x: np.ndarray,
    groups: Sequence[str],
    vector: np.ndarray,
    objective: float,
    gradient_norm: float,
    relative_change: float,
    iterations: int,
    accepted_steps: int,
    step_sizes: Sequence[float],
    h0_scales: Sequence[float],
    line_search_evaluations: int,
    curvature_updates: int,
    curvature_skips: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "ridge_lambda": float(ridge_lambda),
        "training_rows": int(x.shape[0]),
        "training_groups": int(len(set(str(group) for group in groups))),
        "parameter_dimension": int(vector.size),
        "history_size": HISTORY_SIZE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "iterations": int(iterations),
        "accepted_steps": int(accepted_steps),
        "objective": _finite_or_none(float(objective)),
        "gradient_infinity_norm": _finite_or_none(float(gradient_norm)),
        "relative_objective_change": _finite_or_none(float(relative_change)),
        "accepted_step_summary": _numeric_summary(step_sizes),
        "initial_inverse_hessian_scale_summary": _numeric_summary(h0_scales),
        "line_search_evaluations": int(line_search_evaluations),
        "curvature_updates": int(curvature_updates),
        "curvature_skips": int(curvature_skips),
        "coefficient_l2_norm": float(np.linalg.norm(vector)),
        "coefficient_or_prediction_vectors_emitted": False,
    }


def _lbfgs_direction_v2(
    gradient: np.ndarray,
    s_history: Sequence[np.ndarray],
    y_history: Sequence[np.ndarray],
) -> tuple[np.ndarray, float]:
    q = gradient.copy()
    alphas: list[float] = []
    rhos: list[float] = []
    for s_value, y_value in zip(reversed(s_history), reversed(y_history)):
        rho = 1.0 / float(np.dot(y_value, s_value))
        alpha = rho * float(np.dot(s_value, q))
        q -= alpha * y_value
        alphas.append(alpha)
        rhos.append(rho)
    if s_history:
        last_s = s_history[-1]
        last_y = y_history[-1]
        h0_scale = float(
            np.dot(last_s, last_y) / np.dot(last_y, last_y)
        )
        r = h0_scale * q
    else:
        h0_scale = 1.0
        r = q
    for s_value, y_value, alpha, rho in zip(
        s_history, y_history, reversed(alphas), reversed(rhos)
    ):
        beta = rho * float(np.dot(y_value, r))
        r += s_value * (alpha - beta)
    return -r, h0_scale


def fit_lbfgs_v2(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
    initial: np.ndarray | None = None,
    *,
    history_size: int = HISTORY_SIZE,
    maximum_iterations: int = MAXIMUM_ITERATIONS,
    gradient_tolerance: float = GRADIENT_TOLERANCE,
    relative_objective_tolerance: float = RELATIVE_OBJECTIVE_TOLERANCE,
    armijo_c1: float = ARMIJO_C1,
    backtrack_factor: float = BACKTRACK_FACTOR,
) -> OptimizerResultV2:
    """Frozen full-dimension-memory L-BFGS with unchanged Armijo search."""
    dimension = x.shape[1] * target_contrasts.shape[1]
    if dimension != PARAMETER_DIMENSION and not (
        x.shape[1] == 1 and target_contrasts.shape[0] == 6
    ):
        raise base.ModelError(f"unexpected v2 optimizer dimension {dimension}")
    if (
        history_size != HISTORY_SIZE
        or maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
        or armijo_c1 != ARMIJO_C1
        or backtrack_factor != BACKTRACK_FACTOR
    ):
        raise base.ModelError("a caller attempted to override frozen v2 solver settings")
    if initial is None:
        vector = np.zeros(dimension, dtype=float)
    else:
        initial_array = np.asarray(initial, dtype=float)
        if initial_array.shape == (x.shape[1], target_contrasts.shape[1]):
            vector = initial_array.ravel().copy()
        elif initial_array.shape == (dimension,):
            vector = initial_array.copy()
        else:
            raise ValueError("optimizer initial value has the wrong shape")

    value, gradient = base.objective_and_gradient(
        vector, x, y, k_values, groups, target_contrasts, ridge_lambda
    )
    gradient_norm = float(np.max(np.abs(gradient)))
    step_sizes: list[float] = []
    h0_scales: list[float] = []
    line_search_evaluations = 0
    curvature_updates = 0
    curvature_skips = 0
    if gradient_norm <= gradient_tolerance:
        telemetry = _solver_telemetry(
            status="converged_at_initial_value",
            ridge_lambda=ridge_lambda,
            x=x,
            groups=groups,
            vector=vector,
            objective=value,
            gradient_norm=gradient_norm,
            relative_change=0.0,
            iterations=0,
            accepted_steps=0,
            step_sizes=step_sizes,
            h0_scales=h0_scales,
            line_search_evaluations=line_search_evaluations,
            curvature_updates=curvature_updates,
            curvature_skips=curvature_skips,
        )
        FIT_TELEMETRY.append(telemetry)
        return OptimizerResultV2(
            coefficients=vector.reshape(x.shape[1], target_contrasts.shape[1]),
            objective=value,
            gradient_infinity_norm=gradient_norm,
            iterations=0,
            accepted_steps=0,
            relative_objective_change=0.0,
            solver_telemetry=telemetry,
        )

    s_history: list[np.ndarray] = []
    y_history: list[np.ndarray] = []
    accepted_steps = 0
    relative_change = math.inf
    for iteration in range(1, maximum_iterations + 1):
        direction, h0_scale = _lbfgs_direction_v2(
            gradient, s_history, y_history
        )
        h0_scales.append(h0_scale)
        directional_derivative = float(np.dot(gradient, direction))
        if not directional_derivative < 0.0:
            telemetry = _solver_telemetry(
                status="non_descent_direction",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration - 1,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                h0_scales=h0_scales,
                line_search_evaluations=line_search_evaluations,
                curvature_updates=curvature_updates,
                curvature_skips=curvature_skips,
            )
            FIT_TELEMETRY.append(telemetry)
            raise base.ModelError(
                "v2 L-BFGS produced a non-descent direction; telemetry="
                + json.dumps(telemetry, sort_keys=True)
            )

        step = INITIAL_STEP
        accepted: tuple[np.ndarray, float, np.ndarray] | None = None
        for _ in range(LINE_SEARCH_MAXIMUM_EVALUATIONS):
            candidate = vector + step * direction
            candidate_value, candidate_gradient = base.objective_and_gradient(
                candidate,
                x,
                y,
                k_values,
                groups,
                target_contrasts,
                ridge_lambda,
            )
            line_search_evaluations += 1
            if candidate_value <= value + armijo_c1 * step * directional_derivative:
                accepted = (candidate, candidate_value, candidate_gradient)
                break
            step *= backtrack_factor
        if accepted is None:
            telemetry = _solver_telemetry(
                status="line_search_failure",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration - 1,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                h0_scales=h0_scales,
                line_search_evaluations=line_search_evaluations,
                curvature_updates=curvature_updates,
                curvature_skips=curvature_skips,
            )
            FIT_TELEMETRY.append(telemetry)
            raise base.ModelError(
                "v2 deterministic Armijo line search failed; telemetry="
                + json.dumps(telemetry, sort_keys=True)
            )

        new_vector, new_value, new_gradient = accepted
        step_sizes.append(step)
        s_value = new_vector - vector
        y_value = new_gradient - gradient
        curvature = float(np.dot(s_value, y_value))
        curvature_floor = (
            CURVATURE_FLOOR_MULTIPLIER
            * max(1.0, float(np.linalg.norm(s_value)))
            * max(1.0, float(np.linalg.norm(y_value)))
        )
        if curvature > curvature_floor:
            s_history.append(s_value)
            y_history.append(y_value)
            curvature_updates += 1
            if len(s_history) > history_size:
                del s_history[0]
                del y_history[0]
        else:
            curvature_skips += 1

        relative_change = abs(new_value - value) / max(
            1.0, abs(value), abs(new_value)
        )
        vector, value, gradient = new_vector, new_value, new_gradient
        accepted_steps += 1
        gradient_norm = float(np.max(np.abs(gradient)))
        if (
            gradient_norm <= gradient_tolerance
            and relative_change <= relative_objective_tolerance
        ):
            telemetry = _solver_telemetry(
                status="converged",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient_norm=gradient_norm,
                relative_change=relative_change,
                iterations=iteration,
                accepted_steps=accepted_steps,
                step_sizes=step_sizes,
                h0_scales=h0_scales,
                line_search_evaluations=line_search_evaluations,
                curvature_updates=curvature_updates,
                curvature_skips=curvature_skips,
            )
            FIT_TELEMETRY.append(telemetry)
            return OptimizerResultV2(
                coefficients=vector.reshape(
                    x.shape[1], target_contrasts.shape[1]
                ),
                objective=value,
                gradient_infinity_norm=gradient_norm,
                iterations=iteration,
                accepted_steps=accepted_steps,
                relative_objective_change=relative_change,
                solver_telemetry=telemetry,
            )

    telemetry = _solver_telemetry(
        status="iteration_cap_exhausted",
        ridge_lambda=ridge_lambda,
        x=x,
        groups=groups,
        vector=vector,
        objective=value,
        gradient_norm=gradient_norm,
        relative_change=relative_change,
        iterations=maximum_iterations,
        accepted_steps=accepted_steps,
        step_sizes=step_sizes,
        h0_scales=h0_scales,
        line_search_evaluations=line_search_evaluations,
        curvature_updates=curvature_updates,
        curvature_skips=curvature_skips,
    )
    FIT_TELEMETRY.append(telemetry)
    raise base.ModelError(
        "v2 deterministic L-BFGS did not meet both convergence tolerances in "
        f"{maximum_iterations} iterations; telemetry="
        + json.dumps(telemetry, sort_keys=True)
    )


# Every unchanged v1 fitting route resolves this global at call time.
base.fit_lbfgs = fit_lbfgs_v2


def _synthetic_ill_conditioned_stress() -> dict[str, Any]:
    """Exercise all 255 parameters under deterministic collinearity."""
    n = sum(base.GROUP_SIZES.values())
    groups = [
        group for group, size in base.GROUP_SIZES.items() for _ in range(size)
    ]
    t = np.linspace(-1.0, 1.0, n)
    x = np.ones((n, 17), dtype=float)
    for pair in range(8):
        frequency = float(pair + 1)
        common = np.sin(math.pi * frequency * t) + 0.2 * np.cos(
            math.pi * (frequency + 0.5) * t
        )
        perturbation = np.cos(math.pi * (frequency + 2.25) * t)
        common = (common - np.mean(common)) / np.std(common)
        perturbation = (perturbation - np.mean(perturbation)) / np.std(
            perturbation
        )
        x[:, 1 + 2 * pair] = common
        x[:, 2 + 2 * pair] = common + 0.002 * perturbation

    contrasts = base.helmert(16)
    planted = np.zeros((17, 15), dtype=float)
    planted[0] = np.linspace(-0.22, 0.24, 15)
    for pair in range(8):
        target = pair % 15
        planted[1 + 2 * pair, target] = 75.0
        planted[2 + 2 * pair, target] = -75.0
    eta = (x @ planted) @ contrasts.T
    k_values = 1 + (np.arange(n, dtype=int) % 15)
    _, soft_targets = base.batch_fixed_k_logz_marginals(eta, k_values)
    condition_number = float(np.linalg.cond(x))

    FIT_TELEMETRY.clear()
    ridge_warm = fit_lbfgs_v2(
        x, soft_targets, k_values, groups, contrasts, 1.0e-4
    )
    first = fit_lbfgs_v2(
        x,
        soft_targets,
        k_values,
        groups,
        contrasts,
        1.0e-5,
        initial=ridge_warm.coefficients,
    )
    second = fit_lbfgs_v2(
        x,
        soft_targets,
        k_values,
        groups,
        contrasts,
        1.0e-5,
        initial=ridge_warm.coefficients,
    )
    first_prediction = base.predict(
        first.coefficients, x, soft_targets, k_values, contrasts
    )
    second_prediction = base.predict(
        second.coefficients, x, soft_targets, k_values, contrasts
    )
    coefficient_error = float(
        np.max(np.abs(first.coefficients - second.coefficients))
    )
    marginal_error = float(
        np.max(np.abs(first_prediction.marginals - second_prediction.marginals))
    )
    if (
        coefficient_error > REPEAT_COEFFICIENT_TOLERANCE
        or marginal_error > REPEAT_PREDICTION_TOLERANCE
    ):
        raise AssertionError("v2 ill-conditioned synthetic repeat was not deterministic")
    return {
        "status": "pass",
        "rows": n,
        "parameter_dimension": PARAMETER_DIMENSION,
        "design_condition_number": condition_number,
        "ridge_warm_lambda": 1.0e-4,
        "stress_lambda": 1.0e-5,
        "repeat_coefficient_max_absolute_error": coefficient_error,
        "repeat_marginal_max_absolute_error": marginal_error,
        "warm_fit": ridge_warm.solver_telemetry,
        "stress_fit": first.solver_telemetry,
        "uses_real_data": False,
    }


def run_self_test(base_contract_path: Path, addendum_path: Path) -> dict[str, Any]:
    validate_solver_addendum(base_contract_path, addendum_path)
    base_result = base.run_self_test(base_contract_path)
    stress = _synthetic_ill_conditioned_stress()
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v2_synthetic_self_test",
        "base_v1_self_test": base_result,
        "full_dimension_ill_conditioned_stress": stress,
        "final_or_unassigned_opened": False,
        "uses_real_data": False,
    }


def _fit_probe_path(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    *,
    warm_start: bool,
) -> dict[float, tuple[OptimizerResultV2, Any]]:
    contrasts = base.helmert(16)
    initial: np.ndarray | None = None
    path: dict[float, tuple[OptimizerResultV2, Any]] = {}
    for ridge_lambda in DESCENDING_LAMBDAS:
        result = fit_lbfgs_v2(
            x,
            y,
            k_values,
            groups,
            contrasts,
            ridge_lambda,
            initial=initial if warm_start else None,
        )
        prediction = base.predict(
            result.coefficients, x, y, k_values, contrasts
        )
        path[ridge_lambda] = (result, prediction)
        if warm_start:
            initial = result.coefficients
    return path


def _prepare_probe_output(path: Path) -> Path:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise base.ModelError("the real-data solver probe must run in a Slurm job")
    scratch = Path(scratch_value).resolve(strict=True)
    episode_root = (scratch / base.EPISODE_SCRATCH_RELATIVE).resolve(strict=True)
    if not path.is_absolute() or path.is_symlink():
        raise base.ModelError("probe output must be a nonsymlink absolute Scratch path")
    expected_name = f"v3_m0_solver_probe_v2_job_{job_id}"
    if path.name != expected_name:
        raise base.ModelError(f"probe output basename must be {expected_name!r}")
    if path.parent.resolve(strict=True) != episode_root:
        raise base.ModelError("probe output must be a direct child of EP10 Scratch")
    if path.exists():
        resolved = path.resolve(strict=True)
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise base.ModelError("probe output must be absent or empty")
    else:
        path.mkdir(parents=False)
        resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(scratch):
        raise base.ModelError("probe output escaped Scratch")
    return resolved


def _zero_start_objective_tolerance(ridge_lambda: float) -> float:
    strong_convexity_bound = (
        1.1
        * PARAMETER_DIMENSION
        * GRADIENT_TOLERANCE**2
        / (2.0 * ridge_lambda)
    )
    return max(1.0e-10, strong_convexity_bound)


def _compare_probe_paths(
    warm_first: Mapping[float, tuple[OptimizerResultV2, Any]],
    warm_second: Mapping[float, tuple[OptimizerResultV2, Any]],
    zero_start: Mapping[float, tuple[OptimizerResultV2, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repeat_checks: list[dict[str, Any]] = []
    zero_checks: list[dict[str, Any]] = []
    for ridge_lambda in DESCENDING_LAMBDAS:
        first_result, first_prediction = warm_first[ridge_lambda]
        second_result, second_prediction = warm_second[ridge_lambda]
        zero_result, zero_prediction = zero_start[ridge_lambda]

        repeat_objective = abs(first_result.objective - second_result.objective)
        repeat_coefficient = float(
            np.max(np.abs(first_result.coefficients - second_result.coefficients))
        )
        repeat_marginal = float(
            np.max(
                np.abs(first_prediction.marginals - second_prediction.marginals)
            )
        )
        repeat_log_probability = float(
            np.max(
                np.abs(
                    first_prediction.log_probability_nats
                    - second_prediction.log_probability_nats
                )
            )
        )
        repeat_pass = (
            repeat_objective <= REPEAT_OBJECTIVE_TOLERANCE
            and repeat_coefficient <= REPEAT_COEFFICIENT_TOLERANCE
            and repeat_marginal <= REPEAT_PREDICTION_TOLERANCE
            and repeat_log_probability <= REPEAT_PREDICTION_TOLERANCE
        )
        repeat_checks.append(
            {
                "lambda": ridge_lambda,
                "objective_absolute_difference": repeat_objective,
                "coefficient_max_absolute_difference": repeat_coefficient,
                "marginal_max_absolute_difference": repeat_marginal,
                "log_probability_max_absolute_difference": repeat_log_probability,
                "pass": repeat_pass,
            }
        )

        zero_objective = abs(first_result.objective - zero_result.objective)
        zero_coefficient = float(
            np.max(np.abs(first_result.coefficients - zero_result.coefficients))
        )
        zero_marginal = float(
            np.max(np.abs(first_prediction.marginals - zero_prediction.marginals))
        )
        zero_log_probability = float(
            np.max(
                np.abs(
                    first_prediction.log_probability_nats
                    - zero_prediction.log_probability_nats
                )
            )
        )
        objective_tolerance = _zero_start_objective_tolerance(ridge_lambda)
        zero_pass = (
            zero_objective <= objective_tolerance
            and zero_marginal <= ZERO_START_MARGINAL_TOLERANCE
            and zero_log_probability <= ZERO_START_LOG_PROBABILITY_TOLERANCE
        )
        zero_checks.append(
            {
                "lambda": ridge_lambda,
                "objective_absolute_difference": zero_objective,
                "objective_absolute_tolerance": objective_tolerance,
                "coefficient_max_absolute_difference_report_only": zero_coefficient,
                "marginal_max_absolute_difference": zero_marginal,
                "marginal_max_absolute_tolerance": ZERO_START_MARGINAL_TOLERANCE,
                "log_probability_max_absolute_difference": zero_log_probability,
                "log_probability_max_absolute_tolerance": ZERO_START_LOG_PROBABILITY_TOLERANCE,
                "pass": zero_pass,
            }
        )
    return repeat_checks, zero_checks


def run_targeted_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    output_root = _prepare_probe_output(args.output_root)
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v2_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_or_prediction_vectors_emitted": False,
        "target_contrast_matrix_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v2.json"
    try:
        synthetic_stress = _synthetic_ill_conditioned_stress()
        FIT_TELEMETRY.clear()
        records = base.load_real_records(args.target_root, args.interface_root)
        training = [
            record
            for record in records
            if record.group
            not in (PROBE_OUTER_HELD_GROUP, PROBE_INNER_HELD_GROUP)
        ]
        training_groups = sorted(set(record.group for record in training))
        if (
            len(training) != PROBE_EXPECTED_CELLS
            or len(training_groups) != PROBE_EXPECTED_GROUPS
            or PROBE_OUTER_HELD_GROUP in training_groups
            or PROBE_INNER_HELD_GROUP in training_groups
        ):
            raise base.ModelError("targeted probe training split changed")

        transform = base.fit_design_transform(training)
        x = base.build_design(training, transform)
        y, k_values, groups = base.outcome_matrix(training)

        FIT_TELEMETRY.clear()
        warm_first = _fit_probe_path(x, y, k_values, groups, warm_start=True)
        first_telemetry = [
            warm_first[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        warm_second = _fit_probe_path(x, y, k_values, groups, warm_start=True)
        second_telemetry = [
            warm_second[value][0].solver_telemetry
            for value in DESCENDING_LAMBDAS
        ]
        zero_start = _fit_probe_path(x, y, k_values, groups, warm_start=False)
        zero_telemetry = [
            zero_start[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        repeat_checks, zero_checks = _compare_probe_paths(
            warm_first, warm_second, zero_start
        )
        warm_path_worst_iterations = max(
            result.iterations
            for path in (warm_first, warm_second)
            for result, _ in path.values()
        )
        if warm_path_worst_iterations >= WARM_PATH_MAX_ITERATIONS_EXCLUSIVE:
            raise base.ModelError(
                "targeted probe lacks frozen warm-path iteration headroom: "
                f"worst={warm_path_worst_iterations}, required <"
                f"{WARM_PATH_MAX_ITERATIONS_EXCLUSIVE}"
            )
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("targeted probe deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("targeted probe warm-start versus zero-start check failed")

        payload.update(
            {
                "status": "pass",
                "training_cells": len(training),
                "training_groups": len(training_groups),
                "descending_lambda_order": list(DESCENDING_LAMBDAS),
                "synthetic_full_dimension_stress": synthetic_stress,
                "warm_path_repeat_1": first_telemetry,
                "warm_path_repeat_2": second_telemetry,
                "zero_start_path": zero_telemetry,
                "deterministic_repeat_checks": repeat_checks,
                "warm_start_vs_zero_start_checks": zero_checks,
                "all_fits_met_both_unchanged_tolerances": True,
                "warm_path_worst_iterations": warm_path_worst_iterations,
                "warm_path_max_iterations_exclusive": WARM_PATH_MAX_ITERATIONS_EXCLUSIVE,
                "warm_path_iteration_headroom_pass": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        payload.update(
            {
                "status": "fail",
                "error": str(error),
                "completed_fit_telemetry": list(FIT_TELEMETRY),
            }
        )
        base._atomic_json(output_path, payload)
        raise


def run_execute(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    FIT_TELEMETRY.clear()
    summary = base.run_real(args)
    summary["solver_addendum_v2"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v2",
        "history_size": HISTORY_SIZE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "line_search": "unchanged_Armijo_backtracking",
        "convergence_tolerances_changed": False,
    }
    base._atomic_json(args.output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        args.output_root / "v3_m0_solver_telemetry_v2.json",
        {
            "status": "pass",
            "solver_addendum": args.solver_addendum.name,
            "fits": list(FIT_TELEMETRY),
            "coefficient_or_prediction_vectors_emitted": False,
            "final_or_unassigned_opened": False,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument("--solver-addendum", type=Path, default=ADDENDUM_PATH)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temperature-diagnostic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("--self-test does not accept real input or output paths")
        if args.temperature_diagnostic:
            parser.error("--temperature-diagnostic is unavailable in self-test mode")
        try:
            result = run_self_test(args.contract, args.solver_addendum)
        except (base.ModelError, AssertionError, OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    missing = [
        name
        for name in ("target_root", "interface_root", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "selected real-data mode requires: "
            + ", ".join("--" + value.replace("_", "-") for value in missing)
        )
    if args.probe and args.temperature_diagnostic:
        parser.error("the numerical probe cannot run the temperature diagnostic")
    try:
        if args.probe:
            result = run_targeted_probe(args)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "stage": result["stage"],
                        "output_root": result["output_root"],
                    },
                    sort_keys=True,
                )
            )
        else:
            run_execute(args)
    except base.BoundaryReview as error:
        print(f"CONTRACT REVIEW REQUIRED: {error}", file=sys.stderr)
        return 3
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
