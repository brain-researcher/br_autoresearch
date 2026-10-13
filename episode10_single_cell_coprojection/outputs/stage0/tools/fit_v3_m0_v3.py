#!/usr/bin/env python3
"""Run EP10 v3 M0 with the frozen v3 expanding strong-Wolfe solver.

The untouched v1 scientific engine supplies all model, likelihood, design,
resampling, scoring, and sealed-input behavior.  The v3 addendum retains the
v2 full-dimension L-BFGS memory and iteration cap and replaces only its
shrink-only line search with deterministic expanding strong-Wolfe bracketing
and bisection zoom.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
if (
    not SCRIPT_PATH.is_file()
    or SCRIPT_PATH.is_symlink()
    or SCRIPT_PATH != SCRIPT_PATH.resolve(strict=True)
):
    raise ImportError("v3 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V2_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v2.py"
if (
    not V2_ENGINE_PATH.is_file()
    or V2_ENGINE_PATH.is_symlink()
    or V2_ENGINE_PATH != V2_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v2 predecessor wrapper must be an unsymlinked stage0 file")

_V2_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v2_for_v3", V2_ENGINE_PATH
)
if _V2_SPEC is None or _V2_SPEC.loader is None:
    raise ImportError(f"cannot import v2 solver wrapper: {V2_ENGINE_PATH}")
v2 = importlib.util.module_from_spec(_V2_SPEC)
sys.modules[_V2_SPEC.name] = v2
_V2_SPEC.loader.exec_module(v2)
base = v2.base


BASE_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0.py"
BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V2_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v2.yaml"
V3_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v3.yaml"

PARAMETER_DIMENSION = 255
HISTORY_SIZE = 255
MAXIMUM_ITERATIONS = 2000
GRADIENT_TOLERANCE = 1.0e-7
RELATIVE_OBJECTIVE_TOLERANCE = 1.0e-10
ARMIJO_C1 = 1.0e-4
CURVATURE_C2 = 0.9
INITIAL_STEP = 1.0
EXPANSION_FACTOR = 2.0
MAXIMUM_STEP = 1048576.0
LINE_SEARCH_MAXIMUM_EVALUATIONS = 80
ZOOM_MAXIMUM_ITERATIONS = 60
ZOOM_BISECTION_FRACTION = 0.5
CURVATURE_FLOOR_MULTIPLIER = 1.0e-14
DESCENDING_LAMBDAS = tuple(sorted(base.LAMBDAS, reverse=True))

PROBE_OUTER_HELD_GROUP = "212074"
PROBE_INNER_HELD_GROUP = "220187"
PROBE_EXPECTED_CELLS = 153
PROBE_EXPECTED_GROUPS = 10
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = 1000
PRODUCTION_OUTPUT_BASENAME = "v3_m0_fit_v3"

FIT_TELEMETRY: list[dict[str, Any]] = []


@dataclass(frozen=True)
class StrongWolfeResult:
    vector: np.ndarray
    value: float
    gradient: np.ndarray
    step: float
    evaluations: int
    expansions: int
    zoom_iterations: int
    directional_derivative_initial: float
    directional_derivative_accepted: float
    armijo_slack: float
    curvature_ratio: float


@dataclass(frozen=True)
class OptimizerResultV3:
    coefficients: np.ndarray
    objective: float
    gradient_infinity_norm: float
    iterations: int
    accepted_steps: int
    relative_objective_change: float
    solver_telemetry: Mapping[str, Any]


class StrongWolfeFailure(base.ModelError):
    def __init__(
        self,
        message: str,
        *,
        evaluations: int,
        expansions: int,
        zoom_iterations: int,
    ) -> None:
        super().__init__(message)
        self.evaluations = evaluations
        self.expansions = expansions
        self.zoom_iterations = zoom_iterations


def _as_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise base.ModelError(f"v3 solver addendum has invalid {label}") from error
    if not math.isfinite(result):
        raise base.ModelError(f"v3 solver addendum has nonfinite {label}")
    return result


def _require_exact_unsymlinked_file(
    observed: Path, expected: Path, label: str
) -> Path:
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
    return absolute


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate v1, v2, and the v3 numerical-only amendment."""
    _require_exact_unsymlinked_file(
        SCRIPT_PATH, SCRIPT_DIR / "fit_v3_m0_v3.py", "v3 wrapper"
    )
    _require_exact_unsymlinked_file(
        BASE_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0.py", "v1 engine"
    )
    _require_exact_unsymlinked_file(
        V2_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0_v2.py", "v2 predecessor wrapper"
    )
    _require_exact_unsymlinked_file(
        base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"
    )
    _require_exact_unsymlinked_file(
        V2_ADDENDUM_PATH, STAGE0_DIR / "v3_m0_solver_addendum_v2.yaml", "v2 addendum"
    )
    _require_exact_unsymlinked_file(
        addendum_path, V3_ADDENDUM_PATH, "v3 addendum"
    )

    base_contract, v2_addendum = v2.validate_solver_addendum(
        base_contract_path, V2_ADDENDUM_PATH
    )
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v3"
        or addendum.get("status")
        != "frozen_before_v3_numerical_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v3 solver addendum identity or sealed scope changed")

    base_reference = addendum.get("base_contract", {})
    predecessor = addendum.get("predecessor_solver_addendum", {})
    if (
        base_reference.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_model_contract.yaml"
        or base_reference.get("required_schema_version")
        != base_contract.get("schema_version")
        or base_reference.get("required_status") != base_contract.get("status")
        or predecessor.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_solver_addendum_v2.yaml"
        or predecessor.get("required_schema_version")
        != v2_addendum.get("schema_version")
        or predecessor.get("required_status") != v2_addendum.get("status")
    ):
        raise base.ModelError("v3 contract references changed")

    solver = addendum.get("solver", {})
    line_search = solver.get("line_search", {})
    zoom = line_search.get("zoom", {})
    convergence = solver.get("convergence", {})
    curvature = solver.get("curvature", {})
    if (
        solver.get("algorithm") != "deterministic_L_BFGS"
        or int(solver.get("parameter_dimension", -1)) != PARAMETER_DIMENSION
        or int(solver.get("history_size", -1)) != HISTORY_SIZE
        or int(solver.get("maximum_iterations", -1)) != MAXIMUM_ITERATIONS
        or line_search.get("algorithm")
        != "deterministic_safeguarded_expanding_strong_Wolfe"
        or _as_float(line_search.get("initial_step"), "initial step")
        != INITIAL_STEP
        or _as_float(line_search.get("armijo_c1"), "Armijo c1") != ARMIJO_C1
        or _as_float(line_search.get("curvature_c2"), "curvature c2")
        != CURVATURE_C2
        or _as_float(line_search.get("expansion_factor"), "expansion factor")
        != EXPANSION_FACTOR
        or _as_float(line_search.get("maximum_step"), "maximum step")
        != MAXIMUM_STEP
        or int(line_search.get("maximum_evaluations", -1))
        != LINE_SEARCH_MAXIMUM_EVALUATIONS
        or zoom.get("interpolation") != "deterministic_bisection"
        or _as_float(zoom.get("bisection_fraction"), "zoom fraction")
        != ZOOM_BISECTION_FRACTION
        or int(zoom.get("maximum_iterations", -1))
        != ZOOM_MAXIMUM_ITERATIONS
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
        raise base.ModelError("v3 solver settings changed")

    ridge_path = addendum.get("ridge_path", {})
    initialization = ridge_path.get("initialization", {})
    if (
        tuple(float(value) for value in ridge_path.get("ordered_lambdas", ()))
        != DESCENDING_LAMBDAS
        or tuple(float(value) for value in base_contract["estimation"]["lambda_grid"])
        != base.LAMBDAS
        or initialization.get("first_lambda") != "all_zero_coefficients"
        or initialization.get("later_lambdas")
        != "immediately_preceding_converged_coefficients"
        or ridge_path.get("auxiliary_lambdas") != "forbidden"
    ):
        raise base.ModelError("v3 ridge grid or warm-start order changed")

    seals = addendum.get("path_seals", {})
    expected_seals = {
        "wrapper": "outputs/stage0/tools/fit_v3_m0_v3.py",
        "base_engine": "outputs/stage0/tools/fit_v3_m0.py",
        "predecessor_wrapper": "outputs/stage0/tools/fit_v3_m0_v2.py",
        "base_contract": "outputs/stage0/v3_m0_model_contract.yaml",
        "predecessor_addendum": "outputs/stage0/v3_m0_solver_addendum_v2.yaml",
        "v3_addendum": "outputs/stage0/v3_m0_solver_addendum_v3.yaml",
    }
    if (
        any(seals.get(key) != value for key, value in expected_seals.items())
        or seals.get("every_named_file_and_existing_path_component_must_be_unsymlinked")
        is not True
        or seals.get("production_output_basename") != PRODUCTION_OUTPUT_BASENAME
    ):
        raise base.ModelError("v3 exact path seals changed")

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
        or probe.get("every_accepted_step_must_satisfy_both_Wolfe_conditions")
        is not True
        or probe.get("held_out_scoring") != "forbidden"
        or probe.get("coefficient_prediction_or_target_contrast_vector_emission")
        != "forbidden"
        or probe.get("final_or_unassigned_access") != "forbidden"
        or _as_float(
            repeat.get("objective_absolute_tolerance"),
            "repeat objective tolerance",
        )
        != v2.REPEAT_OBJECTIVE_TOLERANCE
        or _as_float(
            repeat.get("coefficient_max_absolute_tolerance"),
            "repeat coefficient tolerance",
        )
        != v2.REPEAT_COEFFICIENT_TOLERANCE
        or _as_float(
            repeat.get("prediction_max_absolute_tolerance"),
            "repeat prediction tolerance",
        )
        != v2.REPEAT_PREDICTION_TOLERANCE
        or _as_float(
            zero_start.get("marginal_max_absolute_tolerance"),
            "zero-start marginal tolerance",
        )
        != v2.ZERO_START_MARGINAL_TOLERANCE
        or _as_float(
            zero_start.get("log_probability_max_absolute_tolerance"),
            "zero-start log-probability tolerance",
        )
        != v2.ZERO_START_LOG_PROBABILITY_TOLERANCE
    ):
        raise base.ModelError("v3 targeted-probe contract changed")
    return base_contract, addendum


def _strong_wolfe_search(
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> StrongWolfeResult:
    """Deterministic expanding bracket and bisection-only strong-Wolfe zoom."""
    if (
        not math.isfinite(float(value))
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(direction))
    ):
        raise StrongWolfeFailure(
            "strong-Wolfe search received a nonfinite initial state",
            evaluations=0,
            expansions=0,
            zoom_iterations=0,
        )
    derivative_initial = float(np.dot(gradient, direction))
    if not math.isfinite(derivative_initial) or not derivative_initial < 0.0:
        raise StrongWolfeFailure(
            "strong-Wolfe search requires a descent direction",
            evaluations=0,
            expansions=0,
            zoom_iterations=0,
        )
    evaluations = 0
    expansions = 0
    zoom_iterations = 0

    def evaluate_step(
        step: float,
    ) -> tuple[np.ndarray, float, np.ndarray, float]:
        nonlocal evaluations
        if evaluations >= LINE_SEARCH_MAXIMUM_EVALUATIONS:
            raise StrongWolfeFailure(
                "strong-Wolfe evaluation ceiling reached",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        candidate = vector + step * direction
        if not np.all(np.isfinite(candidate)):
            raise StrongWolfeFailure(
                "strong-Wolfe candidate vector became nonfinite",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        try:
            candidate_value, candidate_gradient = evaluate(candidate)
        except (base.ModelError, ArithmeticError, ValueError) as error:
            raise StrongWolfeFailure(
                f"strong-Wolfe objective evaluation failed: {error}",
                evaluations=evaluations + 1,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            ) from error
        candidate_gradient = np.asarray(candidate_gradient, dtype=float)
        if (
            not math.isfinite(float(candidate_value))
            or candidate_gradient.shape != gradient.shape
            or not np.all(np.isfinite(candidate_gradient))
        ):
            raise StrongWolfeFailure(
                "strong-Wolfe objective evaluation returned a nonfinite or malformed result",
                evaluations=evaluations + 1,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        candidate_derivative = float(np.dot(candidate_gradient, direction))
        evaluations += 1
        if not math.isfinite(candidate_derivative):
            raise StrongWolfeFailure(
                "strong-Wolfe directional derivative became nonfinite",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        return candidate, candidate_value, candidate_gradient, candidate_derivative

    def accepted(
        step: float,
        candidate: np.ndarray,
        candidate_value: float,
        candidate_gradient: np.ndarray,
        candidate_derivative: float,
    ) -> StrongWolfeResult:
        armijo_bound = value + ARMIJO_C1 * step * derivative_initial
        armijo_slack = armijo_bound - candidate_value
        curvature_ratio = abs(candidate_derivative) / abs(derivative_initial)
        if (
            not math.isfinite(armijo_slack)
            or not math.isfinite(curvature_ratio)
            or armijo_slack < 0.0
            or curvature_ratio > CURVATURE_C2
        ):
            raise StrongWolfeFailure(
                "internal strong-Wolfe acceptance violation",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        return StrongWolfeResult(
            vector=candidate,
            value=candidate_value,
            gradient=candidate_gradient,
            step=step,
            evaluations=evaluations,
            expansions=expansions,
            zoom_iterations=zoom_iterations,
            directional_derivative_initial=derivative_initial,
            directional_derivative_accepted=candidate_derivative,
            armijo_slack=armijo_slack,
            curvature_ratio=curvature_ratio,
        )

    def zoom(
        low: float,
        high: float,
        value_low: float,
        derivative_low: float,
    ) -> StrongWolfeResult:
        nonlocal zoom_iterations
        for _ in range(ZOOM_MAXIMUM_ITERATIONS):
            zoom_iterations += 1
            step = ZOOM_BISECTION_FRACTION * (low + high)
            if step == low or step == high:
                raise StrongWolfeFailure(
                    "strong-Wolfe zoom bracket reached floating-point resolution",
                    evaluations=evaluations,
                    expansions=expansions,
                    zoom_iterations=zoom_iterations,
                )
            candidate, candidate_value, candidate_gradient, candidate_derivative = (
                evaluate_step(step)
            )
            armijo_bound = value + ARMIJO_C1 * step * derivative_initial
            if candidate_value > armijo_bound or candidate_value >= value_low:
                high = step
                continue
            if abs(candidate_derivative) <= CURVATURE_C2 * abs(derivative_initial):
                return accepted(
                    step,
                    candidate,
                    candidate_value,
                    candidate_gradient,
                    candidate_derivative,
                )
            if candidate_derivative * (high - low) >= 0.0:
                high = low
            low = step
            value_low = candidate_value
            derivative_low = candidate_derivative
        raise StrongWolfeFailure(
            "strong-Wolfe zoom iteration ceiling reached",
            evaluations=evaluations,
            expansions=expansions,
            zoom_iterations=zoom_iterations,
        )

    previous_step = 0.0
    previous_value = value
    previous_derivative = derivative_initial
    step = INITIAL_STEP
    iteration = 0
    while True:
        candidate, candidate_value, candidate_gradient, candidate_derivative = (
            evaluate_step(step)
        )
        armijo_bound = value + ARMIJO_C1 * step * derivative_initial
        if candidate_value > armijo_bound or (
            iteration > 0 and candidate_value >= previous_value
        ):
            return zoom(
                previous_step,
                step,
                previous_value,
                previous_derivative,
            )
        if abs(candidate_derivative) <= CURVATURE_C2 * abs(derivative_initial):
            return accepted(
                step,
                candidate,
                candidate_value,
                candidate_gradient,
                candidate_derivative,
            )
        if candidate_derivative >= 0.0:
            return zoom(step, previous_step, candidate_value, candidate_derivative)
        if step >= MAXIMUM_STEP:
            raise StrongWolfeFailure(
                "strong-Wolfe maximum step reached without a bracket",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        previous_step = step
        previous_value = candidate_value
        previous_derivative = candidate_derivative
        step = min(EXPANSION_FACTOR * step, MAXIMUM_STEP)
        expansions += 1
        iteration += 1


def _telemetry(
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
    expansion_counts: Sequence[float],
    zoom_counts: Sequence[float],
    armijo_slacks: Sequence[float],
    curvature_ratios: Sequence[float],
    line_search_failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = v2._solver_telemetry(
        status=status,
        ridge_lambda=ridge_lambda,
        x=x,
        groups=groups,
        vector=vector,
        objective=objective,
        gradient_norm=gradient_norm,
        relative_change=relative_change,
        iterations=iterations,
        accepted_steps=accepted_steps,
        step_sizes=step_sizes,
        h0_scales=h0_scales,
        line_search_evaluations=line_search_evaluations,
        curvature_updates=curvature_updates,
        curvature_skips=curvature_skips,
    )
    payload.update(
        {
            "line_search_algorithm": "deterministic_safeguarded_expanding_strong_Wolfe",
            "line_search_parameters": {
                "c1": ARMIJO_C1,
                "c2": CURVATURE_C2,
                "initial_step": INITIAL_STEP,
                "expansion_factor": EXPANSION_FACTOR,
                "maximum_step": MAXIMUM_STEP,
                "maximum_evaluations": LINE_SEARCH_MAXIMUM_EVALUATIONS,
                "zoom": "deterministic_bisection",
                "zoom_maximum_iterations": ZOOM_MAXIMUM_ITERATIONS,
            },
            "line_search_expansion_count_summary": v2._numeric_summary(
                expansion_counts
            ),
            "line_search_zoom_iteration_summary": v2._numeric_summary(zoom_counts),
            "accepted_steps_greater_than_one": int(
                sum(float(step) > 1.0 for step in step_sizes)
            ),
            "accepted_armijo_slack_summary": v2._numeric_summary(armijo_slacks),
            "accepted_curvature_ratio_summary": v2._numeric_summary(
                curvature_ratios
            ),
            "strong_wolfe_all_accepted_steps_pass": bool(
                all(float(value) >= 0.0 for value in armijo_slacks)
                and all(float(value) <= CURVATURE_C2 for value in curvature_ratios)
                and len(armijo_slacks) == accepted_steps
                and len(curvature_ratios) == accepted_steps
            ),
        }
    )
    if line_search_failure is not None:
        payload["line_search_failure"] = dict(line_search_failure)
    return payload


def fit_lbfgs_v3(
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
) -> OptimizerResultV3:
    """Full-dimension-memory L-BFGS with frozen expanding strong-Wolfe search."""
    dimension = x.shape[1] * target_contrasts.shape[1]
    if dimension != PARAMETER_DIMENSION and not (
        x.shape[1] == 1 and target_contrasts.shape[0] == 6
    ):
        raise base.ModelError(f"unexpected v3 optimizer dimension {dimension}")
    if (
        history_size != HISTORY_SIZE
        or maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise base.ModelError("a caller attempted to override frozen v3 solver settings")
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

    def evaluate(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        return base.objective_and_gradient(
            candidate,
            x,
            y,
            k_values,
            groups,
            target_contrasts,
            ridge_lambda,
        )

    value, gradient = evaluate(vector)
    gradient_norm = float(np.max(np.abs(gradient)))
    step_sizes: list[float] = []
    h0_scales: list[float] = []
    expansion_counts: list[float] = []
    zoom_counts: list[float] = []
    armijo_slacks: list[float] = []
    curvature_ratios: list[float] = []
    line_search_evaluations = 0
    curvature_updates = 0
    curvature_skips = 0

    if gradient_norm <= gradient_tolerance:
        telemetry = _telemetry(
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
            line_search_evaluations=0,
            curvature_updates=0,
            curvature_skips=0,
            expansion_counts=expansion_counts,
            zoom_counts=zoom_counts,
            armijo_slacks=armijo_slacks,
            curvature_ratios=curvature_ratios,
        )
        FIT_TELEMETRY.append(telemetry)
        return OptimizerResultV3(
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
        direction, h0_scale = v2._lbfgs_direction_v2(
            gradient, s_history, y_history
        )
        h0_scales.append(h0_scale)
        try:
            line = _strong_wolfe_search(
                vector, value, gradient, direction, evaluate
            )
        except StrongWolfeFailure as error:
            line_search_evaluations += error.evaluations
            telemetry = _telemetry(
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
                expansion_counts=expansion_counts,
                zoom_counts=zoom_counts,
                armijo_slacks=armijo_slacks,
                curvature_ratios=curvature_ratios,
                line_search_failure={
                    "message": str(error),
                    "evaluations": error.evaluations,
                    "expansions": error.expansions,
                    "zoom_iterations": error.zoom_iterations,
                },
            )
            FIT_TELEMETRY.append(telemetry)
            raise base.ModelError(
                "v3 deterministic strong-Wolfe line search failed; telemetry="
                + json.dumps(telemetry, sort_keys=True)
            ) from error

        line_search_evaluations += line.evaluations
        step_sizes.append(line.step)
        expansion_counts.append(float(line.expansions))
        zoom_counts.append(float(line.zoom_iterations))
        armijo_slacks.append(line.armijo_slack)
        curvature_ratios.append(line.curvature_ratio)
        new_vector, new_value, new_gradient = (
            line.vector,
            line.value,
            line.gradient,
        )

        s_value = new_vector - vector
        y_value = new_gradient - gradient
        pair_curvature = float(np.dot(s_value, y_value))
        curvature_floor = (
            CURVATURE_FLOOR_MULTIPLIER
            * max(1.0, float(np.linalg.norm(s_value)))
            * max(1.0, float(np.linalg.norm(y_value)))
        )
        if pair_curvature > curvature_floor:
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
            telemetry = _telemetry(
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
                expansion_counts=expansion_counts,
                zoom_counts=zoom_counts,
                armijo_slacks=armijo_slacks,
                curvature_ratios=curvature_ratios,
            )
            FIT_TELEMETRY.append(telemetry)
            return OptimizerResultV3(
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

    telemetry = _telemetry(
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
        expansion_counts=expansion_counts,
        zoom_counts=zoom_counts,
        armijo_slacks=armijo_slacks,
        curvature_ratios=curvature_ratios,
    )
    FIT_TELEMETRY.append(telemetry)
    raise base.ModelError(
        "v3 deterministic L-BFGS did not meet both convergence tolerances in "
        f"{maximum_iterations} iterations; telemetry="
        + json.dumps(telemetry, sort_keys=True)
    )


# Reuse only v2's audited data-independent stress/path/comparison scaffolding;
# every optimizer call resolves to the v3 implementation below.
base.fit_lbfgs = fit_lbfgs_v3
v2.fit_lbfgs_v2 = fit_lbfgs_v3
v2.FIT_TELEMETRY = FIT_TELEMETRY


def _strong_wolfe_line_search_self_test() -> dict[str, Any]:
    def expansion_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        residual = float(value[0] - 10.0)
        return 0.005 * residual * residual, np.asarray([0.01 * residual])

    expansion_start = np.asarray([0.0])
    expansion_value, expansion_gradient = expansion_objective(expansion_start)
    expansion_result = _strong_wolfe_search(
        expansion_start,
        expansion_value,
        expansion_gradient,
        -expansion_gradient,
        expansion_objective,
    )
    if (
        expansion_result.step <= 1.0
        or expansion_result.expansions < 1
        or expansion_result.armijo_slack < 0.0
        or expansion_result.curvature_ratio > CURVATURE_C2
    ):
        raise AssertionError("strong-Wolfe synthetic expansion check failed")

    def zoom_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        scalar = float(value[0])
        return 5.0 * scalar * scalar, np.asarray([10.0 * scalar])

    zoom_start = np.asarray([1.0])
    zoom_value, zoom_gradient = zoom_objective(zoom_start)
    zoom_result = _strong_wolfe_search(
        zoom_start,
        zoom_value,
        zoom_gradient,
        -zoom_gradient,
        zoom_objective,
    )
    if (
        zoom_result.zoom_iterations < 1
        or zoom_result.armijo_slack < 0.0
        or zoom_result.curvature_ratio > CURVATURE_C2
    ):
        raise AssertionError("strong-Wolfe synthetic bisection-zoom check failed")

    nonfinite_checks: list[dict[str, Any]] = []
    for label, bad_value in (("nan", math.nan), ("infinity", math.inf)):
        def nonfinite_objective(
            value: np.ndarray, emitted: float = bad_value
        ) -> tuple[float, np.ndarray]:
            return emitted, np.asarray([emitted])

        try:
            _strong_wolfe_search(
                np.asarray([0.0]),
                0.5,
                np.asarray([-1.0]),
                np.asarray([1.0]),
                nonfinite_objective,
            )
        except StrongWolfeFailure as error:
            nonfinite_checks.append(
                {"case": label, "status": "pass", "error": str(error)}
            )
        else:
            raise AssertionError(
                f"strong-Wolfe synthetic {label} result was not rejected"
            )
    return {
        "status": "pass",
        "expansion": {
            "accepted_step": expansion_result.step,
            "expansions": expansion_result.expansions,
            "armijo_slack": expansion_result.armijo_slack,
            "curvature_ratio": expansion_result.curvature_ratio,
        },
        "bisection_zoom": {
            "accepted_step": zoom_result.step,
            "zoom_iterations": zoom_result.zoom_iterations,
            "armijo_slack": zoom_result.armijo_slack,
            "curvature_ratio": zoom_result.curvature_ratio,
        },
        "nonfinite_fail_closed": nonfinite_checks,
    }


def run_self_test(base_contract_path: Path, addendum_path: Path) -> dict[str, Any]:
    validate_solver_addendum(base_contract_path, addendum_path)
    line_search = _strong_wolfe_line_search_self_test()
    base_result = base.run_self_test(base_contract_path)
    stress = v2._synthetic_ill_conditioned_stress()
    for label in ("warm_fit", "stress_fit"):
        if stress[label].get("strong_wolfe_all_accepted_steps_pass") is not True:
            raise AssertionError(f"full-dimension stress {label} violated strong Wolfe")
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v3_synthetic_self_test",
        "strong_wolfe_line_search": line_search,
        "base_v1_self_test": base_result,
        "full_dimension_ill_conditioned_stress": stress,
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }


def _require_unsymlinked_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise base.ModelError(f"{label} must be absolute")
    absolute = Path(os.path.abspath(path))
    if (
        not absolute.is_dir()
        or absolute.is_symlink()
        or absolute != absolute.resolve(strict=True)
    ):
        raise base.ModelError(f"{label} must be an unsymlinked directory tree")
    return absolute


def _validate_real_input_roots(
    target_root: Path, interface_root: Path
) -> tuple[Path, Path, Path]:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise base.ModelError("SCRATCH is unset")
    scratch_path = Path(scratch_value)
    scratch = _require_unsymlinked_directory(scratch_path, "SCRATCH")
    episode_root = _require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    expected_target = episode_root / base.TARGET_ROOT_BASENAME
    expected_interface = episode_root / base.INTERFACE_ROOT_BASENAME
    observed_target = _require_unsymlinked_directory(target_root, "target root")
    observed_interface = _require_unsymlinked_directory(
        interface_root, "interface root"
    )
    if observed_target != expected_target or observed_interface != expected_interface:
        raise base.ModelError("real input roots are not the exact sealed Scratch roots")
    for path, label in (
        (observed_target / "primary_16_target_calls.csv", "target calls"),
        (observed_target / "v3_gate_summary.json", "target gate"),
        (observed_interface / "v3_model_interface.csv", "model interface"),
        (
            observed_interface / "v3_model_interface_summary.json",
            "interface gate",
        ),
    ):
        absolute = Path(os.path.abspath(path))
        if (
            not absolute.is_file()
            or absolute.is_symlink()
            or absolute != absolute.resolve(strict=True)
        ):
            raise base.ModelError(f"{label} must be an unsymlinked sealed input")
    return scratch, observed_target, observed_interface


def _prepare_unique_output(
    path: Path, scratch: Path, *, probe: bool
) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real v3 solver work must run in a Slurm job")
    episode_root = _require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    if not path.is_absolute():
        raise base.ModelError("output root must be absolute")
    absolute = Path(os.path.abspath(path))
    expected_name = (
        f"v3_m0_solver_probe_v3_job_{job_id}"
        if probe
        else PRODUCTION_OUTPUT_BASENAME
    )
    if absolute.parent != episode_root or absolute.name != expected_name:
        raise base.ModelError(
            f"output must be the exact EP10 Scratch child {expected_name!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("v3 output root must be absent")
    if probe:
        absolute.mkdir(parents=False)
        if absolute != absolute.resolve(strict=True) or absolute.is_symlink():
            raise base.ModelError("probe output root failed its path seal")
    return absolute


def run_targeted_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = _validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(args.output_root, scratch, probe=True)
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v3_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v3.json"
    try:
        line_search_test = _strong_wolfe_line_search_self_test()
        synthetic_stress = v2._synthetic_ill_conditioned_stress()
        FIT_TELEMETRY.clear()

        records = base.load_real_records(target_root, interface_root)
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
            raise base.ModelError("targeted v3 probe training split changed")

        transform = base.fit_design_transform(training)
        x = base.build_design(training, transform)
        y, k_values, groups = base.outcome_matrix(training)

        FIT_TELEMETRY.clear()
        warm_first = v2._fit_probe_path(x, y, k_values, groups, warm_start=True)
        first_telemetry = [
            warm_first[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        warm_second = v2._fit_probe_path(x, y, k_values, groups, warm_start=True)
        second_telemetry = [
            warm_second[value][0].solver_telemetry
            for value in DESCENDING_LAMBDAS
        ]
        zero_start = v2._fit_probe_path(x, y, k_values, groups, warm_start=False)
        zero_telemetry = [
            zero_start[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        repeat_checks, zero_checks = v2._compare_probe_paths(
            warm_first, warm_second, zero_start
        )
        warm_path_worst_iterations = max(
            result.iterations
            for path in (warm_first, warm_second)
            for result, _ in path.values()
        )
        all_results = [
            result
            for path in (warm_first, warm_second, zero_start)
            for result, _ in path.values()
        ]
        if warm_path_worst_iterations >= WARM_PATH_MAX_ITERATIONS_EXCLUSIVE:
            raise base.ModelError(
                "v3 probe lacks frozen warm-path iteration headroom: "
                f"worst={warm_path_worst_iterations}, required <"
                f"{WARM_PATH_MAX_ITERATIONS_EXCLUSIVE}"
            )
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in all_results
        ):
            raise base.ModelError("a v3 probe fit accepted a non-Wolfe step")
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("v3 probe deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("v3 probe warm-start versus zero-start check failed")

        payload.update(
            {
                "status": "pass",
                "training_cells": len(training),
                "training_groups": len(training_groups),
                "descending_lambda_order": list(DESCENDING_LAMBDAS),
                "strong_wolfe_synthetic_checks": line_search_test,
                "synthetic_full_dimension_stress": synthetic_stress,
                "warm_path_repeat_1": first_telemetry,
                "warm_path_repeat_2": second_telemetry,
                "zero_start_path": zero_telemetry,
                "deterministic_repeat_checks": repeat_checks,
                "warm_start_vs_zero_start_checks": zero_checks,
                "warm_path_worst_iterations": warm_path_worst_iterations,
                "warm_path_max_iterations_exclusive": WARM_PATH_MAX_ITERATIONS_EXCLUSIVE,
                "warm_path_iteration_headroom_pass": True,
                "all_fits_met_both_unchanged_tolerances": True,
                "all_accepted_steps_satisfied_both_Wolfe_conditions": True,
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
    scratch, target_root, interface_root = _validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(args.output_root, scratch, probe=False)
    args.target_root = target_root
    args.interface_root = interface_root
    args.output_root = output_root
    FIT_TELEMETRY.clear()
    summary = base.run_real(args)
    summary["solver_addendum_v3"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v3",
        "history_size": HISTORY_SIZE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "line_search": "deterministic_safeguarded_expanding_strong_Wolfe",
        "convergence_tolerances_changed": False,
    }
    base._atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        output_root / "v3_m0_solver_telemetry_v3.json",
        {
            "status": "pass",
            "solver_addendum": args.solver_addendum.name,
            "fits": list(FIT_TELEMETRY),
            "coefficient_prediction_or_target_contrast_vectors_emitted": False,
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
    parser.add_argument("--solver-addendum", type=Path, default=V3_ADDENDUM_PATH)
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
