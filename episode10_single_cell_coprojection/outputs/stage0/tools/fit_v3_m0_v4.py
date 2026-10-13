#!/usr/bin/env python3
"""Run EP10 v3 M0 with the frozen v4 strong-Wolfe predicate order.

The untouched v1 scientific engine and v3 numerical wrapper remain the
authorities for the model and all solver constants.  V4 changes only control
order: after Armijo passes, curvature acceptance precedes nondecrease and
bracketing heuristics in both zoom and the outer expansion loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
if (
    not SCRIPT_PATH.is_file()
    or SCRIPT_PATH.is_symlink()
    or SCRIPT_PATH != SCRIPT_PATH.resolve(strict=True)
):
    raise ImportError("v4 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V3_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v3.py"
if (
    not V3_ENGINE_PATH.is_file()
    or V3_ENGINE_PATH.is_symlink()
    or V3_ENGINE_PATH != V3_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v3 predecessor wrapper must be an unsymlinked stage0 file")

_V3_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v3_for_v4", V3_ENGINE_PATH
)
if _V3_SPEC is None or _V3_SPEC.loader is None:
    raise ImportError(f"cannot import v3 solver wrapper: {V3_ENGINE_PATH}")
v3 = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = v3
_V3_SPEC.loader.exec_module(v3)
base = v3.base
v2 = v3.v2


BASE_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0.py"
BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V3_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v3.yaml"
V4_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v4.yaml"

PARAMETER_DIMENSION = v3.PARAMETER_DIMENSION
HISTORY_SIZE = v3.HISTORY_SIZE
MAXIMUM_ITERATIONS = v3.MAXIMUM_ITERATIONS
GRADIENT_TOLERANCE = v3.GRADIENT_TOLERANCE
RELATIVE_OBJECTIVE_TOLERANCE = v3.RELATIVE_OBJECTIVE_TOLERANCE
ARMIJO_C1 = v3.ARMIJO_C1
CURVATURE_C2 = v3.CURVATURE_C2
INITIAL_STEP = v3.INITIAL_STEP
EXPANSION_FACTOR = v3.EXPANSION_FACTOR
MAXIMUM_STEP = v3.MAXIMUM_STEP
LINE_SEARCH_MAXIMUM_EVALUATIONS = v3.LINE_SEARCH_MAXIMUM_EVALUATIONS
ZOOM_MAXIMUM_ITERATIONS = v3.ZOOM_MAXIMUM_ITERATIONS
ZOOM_BISECTION_FRACTION = v3.ZOOM_BISECTION_FRACTION
CURVATURE_FLOOR_MULTIPLIER = v3.CURVATURE_FLOOR_MULTIPLIER
DESCENDING_LAMBDAS = v3.DESCENDING_LAMBDAS

PROBE_OUTER_HELD_GROUP = v3.PROBE_OUTER_HELD_GROUP
PROBE_INNER_HELD_GROUP = v3.PROBE_INNER_HELD_GROUP
PROBE_EXPECTED_CELLS = v3.PROBE_EXPECTED_CELLS
PROBE_EXPECTED_GROUPS = v3.PROBE_EXPECTED_GROUPS
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = v3.WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
PRODUCTION_OUTPUT_BASENAME = "v3_m0_fit_v4"
FIT_TELEMETRY = v3.FIT_TELEMETRY

# Retain the exact predecessor routine for regression sentinels before the
# module-global optimizer hook is replaced below.
V3_STRONG_WOLFE_ORIGINAL = v3._strong_wolfe_search


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


def _as_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise base.ModelError(f"v4 solver addendum has invalid {label}") from error
    if not math.isfinite(result):
        raise base.ModelError(f"v4 solver addendum has nonfinite {label}")
    return result


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate the v1 scientific contract and v3/v4 numerical amendments."""
    _require_exact_unsymlinked_file(
        SCRIPT_PATH, SCRIPT_DIR / "fit_v3_m0_v4.py", "v4 wrapper"
    )
    _require_exact_unsymlinked_file(
        V3_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0_v3.py", "v3 predecessor wrapper"
    )
    _require_exact_unsymlinked_file(
        BASE_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0.py", "v1 engine"
    )
    _require_exact_unsymlinked_file(
        base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"
    )
    _require_exact_unsymlinked_file(
        V3_ADDENDUM_PATH,
        STAGE0_DIR / "v3_m0_solver_addendum_v3.yaml",
        "v3 predecessor addendum",
    )
    _require_exact_unsymlinked_file(
        addendum_path, V4_ADDENDUM_PATH, "v4 addendum"
    )

    base_contract, v3_addendum = v3.validate_solver_addendum(
        base_contract_path, V3_ADDENDUM_PATH
    )
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v4"
        or addendum.get("status")
        != "frozen_before_v4_numerical_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v4 solver addendum identity or sealed scope changed")

    base_reference = addendum.get("base_contract", {})
    predecessor = addendum.get("predecessor_solver_addendum", {})
    if (
        base_reference.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_model_contract.yaml"
        or base_reference.get("required_schema_version")
        != base_contract.get("schema_version")
        or base_reference.get("required_status") != base_contract.get("status")
        or predecessor.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_solver_addendum_v3.yaml"
        or predecessor.get("required_schema_version")
        != v3_addendum.get("schema_version")
        or predecessor.get("required_status") != v3_addendum.get("status")
    ):
        raise base.ModelError("v4 contract references changed")

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
        or line_search.get("predicate_order")
        != "Armijo_then_curvature_then_bracketing_heuristics"
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
        raise base.ModelError("v4 solver settings changed")

    expected_acceptance = [
        "phi_alpha <= phi_0 + c1 * alpha * directional_derivative_0",
        "abs(directional_derivative_alpha) <= c2 * abs(directional_derivative_0)",
    ]
    if line_search.get("acceptance_requires") != expected_acceptance:
        raise base.ModelError("v4 exact strong-Wolfe acceptance rules changed")

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
        raise base.ModelError("v4 ridge grid or warm-start order changed")

    seals = addendum.get("path_seals", {})
    expected_seals = {
        "wrapper": "outputs/stage0/tools/fit_v3_m0_v4.py",
        "base_engine": "outputs/stage0/tools/fit_v3_m0.py",
        "predecessor_wrapper": "outputs/stage0/tools/fit_v3_m0_v3.py",
        "base_contract": "outputs/stage0/v3_m0_model_contract.yaml",
        "predecessor_addendum": "outputs/stage0/v3_m0_solver_addendum_v3.yaml",
        "v4_addendum": "outputs/stage0/v3_m0_solver_addendum_v4.yaml",
    }
    if (
        any(seals.get(key) != value for key, value in expected_seals.items())
        or seals.get("every_named_file_and_existing_path_component_must_be_unsymlinked")
        is not True
        or seals.get("production_output_basename") != PRODUCTION_OUTPUT_BASENAME
    ):
        raise base.ModelError("v4 exact path seals changed")

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
        raise base.ModelError("v4 targeted-probe contract changed")

    telemetry = addendum.get("telemetry", {})
    self_test = addendum.get("self_test", {})
    if (
        telemetry.get("scalar_only") is not True
        or telemetry.get("coefficients_prediction_vectors_or_target_contrasts")
        != "forbidden"
        or self_test.get("retain_v1_exact_DP_gradient_optimizer_recovery_and_fold_checks")
        is not True
        or self_test.get("retain_v2_255_parameter_ill_conditioned_stress")
        is not True
        or self_test.get("retain_v3_expansion_zoom_and_nonfinite_checks")
        is not True
        or self_test.get("require_rounded_baseline_zoom_regression") is not True
        or self_test.get("require_rounded_outer_expansion_regression") is not True
    ):
        raise base.ModelError("v4 telemetry or self-test seal changed")
    return base_contract, addendum


def _strong_wolfe_search_v4(
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> v3.StrongWolfeResult:
    """V3 search with only Wolfe-predicate/control ordering corrected."""
    if (
        not math.isfinite(float(value))
        or not np.all(np.isfinite(vector))
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(direction))
    ):
        raise v3.StrongWolfeFailure(
            "strong-Wolfe search received a nonfinite initial state",
            evaluations=0,
            expansions=0,
            zoom_iterations=0,
        )
    derivative_initial = float(np.dot(gradient, direction))
    if not math.isfinite(derivative_initial) or not derivative_initial < 0.0:
        raise v3.StrongWolfeFailure(
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
            raise v3.StrongWolfeFailure(
                "strong-Wolfe evaluation ceiling reached",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        candidate = vector + step * direction
        if not np.all(np.isfinite(candidate)):
            raise v3.StrongWolfeFailure(
                "strong-Wolfe candidate vector became nonfinite",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        try:
            candidate_value, candidate_gradient = evaluate(candidate)
        except (base.ModelError, ArithmeticError, ValueError) as error:
            raise v3.StrongWolfeFailure(
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
            raise v3.StrongWolfeFailure(
                "strong-Wolfe objective evaluation returned a nonfinite or malformed result",
                evaluations=evaluations + 1,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        candidate_derivative = float(np.dot(candidate_gradient, direction))
        evaluations += 1
        if not math.isfinite(candidate_derivative):
            raise v3.StrongWolfeFailure(
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
    ) -> v3.StrongWolfeResult:
        armijo_bound = value + ARMIJO_C1 * step * derivative_initial
        armijo_slack = armijo_bound - candidate_value
        curvature_ratio = abs(candidate_derivative) / abs(derivative_initial)
        if (
            not math.isfinite(armijo_slack)
            or not math.isfinite(curvature_ratio)
            or armijo_slack < 0.0
            or curvature_ratio > CURVATURE_C2
        ):
            raise v3.StrongWolfeFailure(
                "internal strong-Wolfe acceptance violation",
                evaluations=evaluations,
                expansions=expansions,
                zoom_iterations=zoom_iterations,
            )
        return v3.StrongWolfeResult(
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
    ) -> v3.StrongWolfeResult:
        nonlocal zoom_iterations
        for _ in range(ZOOM_MAXIMUM_ITERATIONS):
            zoom_iterations += 1
            step = ZOOM_BISECTION_FRACTION * (low + high)
            if step == low or step == high:
                raise v3.StrongWolfeFailure(
                    "strong-Wolfe zoom bracket reached floating-point resolution",
                    evaluations=evaluations,
                    expansions=expansions,
                    zoom_iterations=zoom_iterations,
                )
            candidate, candidate_value, candidate_gradient, candidate_derivative = (
                evaluate_step(step)
            )
            armijo_bound = value + ARMIJO_C1 * step * derivative_initial
            if candidate_value > armijo_bound:
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
            if candidate_value >= value_low:
                high = step
                continue
            if candidate_derivative * (high - low) >= 0.0:
                high = low
            low = step
            value_low = candidate_value
            derivative_low = candidate_derivative
        raise v3.StrongWolfeFailure(
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
        if candidate_value > armijo_bound:
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
        if iteration > 0 and candidate_value >= previous_value:
            return zoom(
                previous_step,
                step,
                previous_value,
                previous_derivative,
            )
        if candidate_derivative >= 0.0:
            return zoom(step, previous_step, candidate_value, candidate_derivative)
        if step >= MAXIMUM_STEP:
            raise v3.StrongWolfeFailure(
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


# V3's L-BFGS implementation resolves this module global on every line search.
# No optimizer behavior besides the routine above is replaced.
v3._strong_wolfe_search = _strong_wolfe_search_v4
base.fit_lbfgs = v3.fit_lbfgs_v3
v2.fit_lbfgs_v2 = v3.fit_lbfgs_v3
v2.FIT_TELEMETRY = FIT_TELEMETRY
fit_lbfgs_v4 = v3.fit_lbfgs_v3


def _expect_v3_failure(
    label: str,
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    objective: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> dict[str, Any]:
    try:
        V3_STRONG_WOLFE_ORIGINAL(
            vector, value, gradient, direction, objective
        )
    except v3.StrongWolfeFailure as error:
        return {
            "status": "pass",
            "case": label,
            "v3_failure": str(error),
            "v3_evaluations": error.evaluations,
            "v3_expansions": error.expansions,
            "v3_zoom_iterations": error.zoom_iterations,
        }
    raise AssertionError(f"v3 unexpectedly accepted rounded {label} sentinel")


def _predicate_order_self_test() -> dict[str, Any]:
    rounded_zoom_baseline = 1.0e16

    def rounded_zoom_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        scalar = float(value[0])
        residual = scalar - 0.5
        return rounded_zoom_baseline + residual * residual, np.asarray(
            [2.0 * residual]
        )

    zoom_start = np.asarray([0.0])
    zoom_value, zoom_gradient = rounded_zoom_objective(zoom_start)
    zoom_direction = -zoom_gradient
    zoom_v3 = _expect_v3_failure(
        "zoom",
        zoom_start,
        zoom_value,
        zoom_gradient,
        zoom_direction,
        rounded_zoom_objective,
    )
    zoom_v4 = _strong_wolfe_search_v4(
        zoom_start,
        zoom_value,
        zoom_gradient,
        zoom_direction,
        rounded_zoom_objective,
    )
    if (
        zoom_v4.step != 0.5
        or zoom_v4.zoom_iterations != 1
        or zoom_v4.armijo_slack != 0.0
        or zoom_v4.curvature_ratio != 0.0
    ):
        raise AssertionError("v4 rounded-baseline zoom regression failed")

    rounded_outer_baseline = float(2**55)

    def rounded_outer_objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        scalar = float(value[0])
        return (
            rounded_outer_baseline + scalar * scalar / 36.0 - scalar,
            np.asarray([scalar / 18.0 - 1.0]),
        )

    outer_start = np.asarray([0.0])
    outer_value, outer_gradient = rounded_outer_objective(outer_start)
    outer_direction = -outer_gradient
    outer_v3 = _expect_v3_failure(
        "outer_expansion",
        outer_start,
        outer_value,
        outer_gradient,
        outer_direction,
        rounded_outer_objective,
    )
    outer_v4 = _strong_wolfe_search_v4(
        outer_start,
        outer_value,
        outer_gradient,
        outer_direction,
        rounded_outer_objective,
    )
    if (
        outer_v4.step != 2.0
        or outer_v4.expansions != 1
        or outer_v4.zoom_iterations != 0
        or outer_v4.armijo_slack != 0.0
        or outer_v4.curvature_ratio > CURVATURE_C2
    ):
        raise AssertionError("v4 rounded outer-expansion regression failed")

    negative_controls: list[dict[str, Any]] = []
    try:
        _strong_wolfe_search_v4(
            np.asarray([0.0]),
            0.0,
            np.asarray([1.0]),
            np.asarray([0.0]),
            lambda point: (0.0, np.asarray([1.0])),
        )
    except v3.StrongWolfeFailure as error:
        negative_controls.append(
            {"case": "non_descent", "status": "pass", "error": str(error)}
        )
    else:
        raise AssertionError("v4 accepted a non-descent direction")

    try:
        _strong_wolfe_search_v4(
            np.asarray([0.0]),
            0.5,
            np.asarray([-1.0]),
            np.asarray([1.0]),
            lambda point: (0.0, np.asarray([0.0, 0.0])),
        )
    except v3.StrongWolfeFailure as error:
        negative_controls.append(
            {"case": "malformed_gradient", "status": "pass", "error": str(error)}
        )
    else:
        raise AssertionError("v4 accepted a malformed gradient")

    return {
        "status": "pass",
        "rounded_baseline_zoom": {
            **zoom_v3,
            "v4_accepted_step": zoom_v4.step,
            "v4_armijo_slack": zoom_v4.armijo_slack,
            "v4_curvature_ratio": zoom_v4.curvature_ratio,
            "v4_zoom_iterations": zoom_v4.zoom_iterations,
        },
        "rounded_outer_expansion": {
            **outer_v3,
            "v4_accepted_step": outer_v4.step,
            "v4_armijo_slack": outer_v4.armijo_slack,
            "v4_curvature_ratio": outer_v4.curvature_ratio,
            "v4_expansions": outer_v4.expansions,
            "v4_zoom_iterations": outer_v4.zoom_iterations,
        },
        "negative_controls": negative_controls,
    }


def _line_search_self_test() -> dict[str, Any]:
    return {
        "status": "pass",
        "retained_v3_checks": v3._strong_wolfe_line_search_self_test(),
        "v4_predicate_order_regressions": _predicate_order_self_test(),
    }


def run_self_test(base_contract_path: Path, addendum_path: Path) -> dict[str, Any]:
    validate_solver_addendum(base_contract_path, addendum_path)
    line_search = _line_search_self_test()
    base_result = base.run_self_test(base_contract_path)
    stress = v2._synthetic_ill_conditioned_stress()
    for label in ("warm_fit", "stress_fit"):
        if stress[label].get("strong_wolfe_all_accepted_steps_pass") is not True:
            raise AssertionError(f"full-dimension stress {label} violated strong Wolfe")
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v4_synthetic_self_test",
        "strong_wolfe_line_search": line_search,
        "base_v1_self_test": base_result,
        "full_dimension_ill_conditioned_stress": stress,
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }


def _prepare_unique_output(path: Path, scratch: Path, *, probe: bool) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real v4 solver work must run in a Slurm job")
    episode_root = v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    if not path.is_absolute():
        raise base.ModelError("output root must be absolute")
    absolute = Path(os.path.abspath(path))
    expected_name = (
        f"v3_m0_solver_probe_v4_job_{job_id}"
        if probe
        else PRODUCTION_OUTPUT_BASENAME
    )
    if absolute.parent != episode_root or absolute.name != expected_name:
        raise base.ModelError(
            f"output must be the exact EP10 Scratch child {expected_name!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("v4 output root must be absent")
    if probe:
        absolute.mkdir(parents=False)
        if absolute != absolute.resolve(strict=True) or absolute.is_symlink():
            raise base.ModelError("probe output root failed its path seal")
    return absolute


def run_targeted_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(args.output_root, scratch, probe=True)
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v4_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v4.json"
    try:
        line_search_test = _line_search_self_test()
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
            raise base.ModelError("targeted v4 probe training split changed")

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
            warm_second[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
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
                "v4 probe lacks frozen warm-path iteration headroom: "
                f"worst={warm_path_worst_iterations}, required <"
                f"{WARM_PATH_MAX_ITERATIONS_EXCLUSIVE}"
            )
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in all_results
        ):
            raise base.ModelError("a v4 probe fit accepted a non-Wolfe step")
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("v4 probe deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("v4 probe warm-start versus zero-start check failed")

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
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(args.output_root, scratch, probe=False)
    args.target_root = target_root
    args.interface_root = interface_root
    args.output_root = output_root
    FIT_TELEMETRY.clear()
    summary = base.run_real(args)
    summary["solver_addendum_v4"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v4",
        "history_size": HISTORY_SIZE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "line_search": "deterministic_safeguarded_expanding_strong_Wolfe",
        "predicate_order": "Armijo_then_curvature_then_bracketing_heuristics",
        "convergence_tolerances_changed": False,
    }
    base._atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        output_root / "v3_m0_solver_telemetry_v4.json",
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
    parser.add_argument("--solver-addendum", type=Path, default=V4_ADDENDUM_PATH)
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
