#!/usr/bin/env python3
"""Run EP10 v3 M0 with the v5 strong-convexity stall certificate.

The v4 line search is called without modification.  V5 changes only the
termination policy after either of two all-finite zoom failures: the current
iterate may be returned unchanged when the frozen gradient tolerance and an
upward-rounded full-ridge strong-convexity gap bound both pass.
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
    raise ImportError("v5 solver wrapper must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V4_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v4.py"
if (
    not V4_ENGINE_PATH.is_file()
    or V4_ENGINE_PATH.is_symlink()
    or V4_ENGINE_PATH != V4_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v4 predecessor wrapper must be an unsymlinked stage0 file")

_V4_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v4_for_v5", V4_ENGINE_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError(f"cannot import v4 solver wrapper: {V4_ENGINE_PATH}")
v4 = importlib.util.module_from_spec(_V4_SPEC)
sys.modules[_V4_SPEC.name] = v4
_V4_SPEC.loader.exec_module(v4)
v3 = v4.v3
v2 = v4.v2
base = v4.base


BASE_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0.py"
BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V4_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v4.yaml"
V5_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v5.yaml"

PARAMETER_DIMENSION = v4.PARAMETER_DIMENSION
HISTORY_SIZE = v4.HISTORY_SIZE
MAXIMUM_ITERATIONS = v4.MAXIMUM_ITERATIONS
GRADIENT_TOLERANCE = v4.GRADIENT_TOLERANCE
RELATIVE_OBJECTIVE_TOLERANCE = v4.RELATIVE_OBJECTIVE_TOLERANCE
ARMIJO_C1 = v4.ARMIJO_C1
CURVATURE_C2 = v4.CURVATURE_C2
INITIAL_STEP = v4.INITIAL_STEP
EXPANSION_FACTOR = v4.EXPANSION_FACTOR
MAXIMUM_STEP = v4.MAXIMUM_STEP
LINE_SEARCH_MAXIMUM_EVALUATIONS = v4.LINE_SEARCH_MAXIMUM_EVALUATIONS
ZOOM_MAXIMUM_ITERATIONS = v4.ZOOM_MAXIMUM_ITERATIONS
ZOOM_BISECTION_FRACTION = v4.ZOOM_BISECTION_FRACTION
CURVATURE_FLOOR_MULTIPLIER = v4.CURVATURE_FLOOR_MULTIPLIER
DESCENDING_LAMBDAS = v4.DESCENDING_LAMBDAS

PROBE_OUTER_HELD_GROUP = v4.PROBE_OUTER_HELD_GROUP
PROBE_INNER_HELD_GROUP = v4.PROBE_INNER_HELD_GROUP
PROBE_EXPECTED_CELLS = v4.PROBE_EXPECTED_CELLS
PROBE_EXPECTED_GROUPS = v4.PROBE_EXPECTED_GROUPS
WARM_PATH_MAX_ITERATIONS_EXCLUSIVE = v4.WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
PRODUCTION_OUTPUT_BASENAME = "v3_m0_fit_v5"

CERTIFICATE_PROBE_LAMBDA = 1000.0
CERTIFICATE_PROBE_EXPECTED_STEPS = (math.ldexp(1.0, -10), 1.0)
CERTIFICATE_PROBE_FAILURE_EVALUATIONS = 61
CERTIFICATE_PROBE_FAILURE_ZOOM_ITERATIONS = 60

ELIGIBLE_FAILURE_MESSAGES = {
    "strong-Wolfe zoom iteration ceiling reached": "zoom_iteration_ceiling",
    "strong-Wolfe zoom bracket reached floating-point resolution": (
        "bracket_floating_resolution"
    ),
}

# This is an immutable reference to the v4 implementation.  V5 never replaces
# or wraps the line-search routine itself.
V4_STRONG_WOLFE_SEARCH = v4._strong_wolfe_search_v4
V4_FIT_LBFGS_ORIGINAL = v4.fit_lbfgs_v4

FIT_TELEMETRY: list[dict[str, Any]] = []


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
        raise base.ModelError(f"v5 solver addendum has invalid {label}") from error
    if not math.isfinite(result):
        raise base.ModelError(f"v5 solver addendum has nonfinite {label}")
    return result


def validate_solver_addendum(
    base_contract_path: Path, addendum_path: Path
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate only the scientific and numerical seals needed by V5."""
    _require_exact_unsymlinked_file(
        SCRIPT_PATH, SCRIPT_DIR / "fit_v3_m0_v5.py", "v5 wrapper"
    )
    _require_exact_unsymlinked_file(
        V4_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0_v4.py", "v4 predecessor wrapper"
    )
    _require_exact_unsymlinked_file(
        BASE_ENGINE_PATH, SCRIPT_DIR / "fit_v3_m0.py", "v1 engine"
    )
    _require_exact_unsymlinked_file(
        base_contract_path, BASE_CONTRACT_PATH, "v1 M0 model contract"
    )
    _require_exact_unsymlinked_file(
        V4_ADDENDUM_PATH,
        STAGE0_DIR / "v3_m0_solver_addendum_v4.yaml",
        "v4 predecessor addendum",
    )
    _require_exact_unsymlinked_file(
        addendum_path, V5_ADDENDUM_PATH, "v5 addendum"
    )

    base_contract, v4_addendum = v4.validate_solver_addendum(
        base_contract_path, V4_ADDENDUM_PATH
    )
    addendum = base._load_yaml(addendum_path)
    if (
        addendum.get("schema_version")
        != "episode10_v3_m0_solver_addendum_v5"
        or addendum.get("status")
        != "frozen_before_v5_certificate_probe_and_execution"
        or addendum.get("scientific_authority") is not False
        or addendum.get("scope", {}).get("role") != "development_only"
        or addendum.get("scope", {}).get("final_or_unassigned_access")
        != "forbidden"
    ):
        raise base.ModelError("v5 addendum identity or sealed scope changed")

    base_reference = addendum.get("base_contract", {})
    predecessor = addendum.get("predecessor_solver_addendum", {})
    if (
        base_reference.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_model_contract.yaml"
        or base_reference.get("required_schema_version")
        != base_contract.get("schema_version")
        or base_reference.get("required_status") != base_contract.get("status")
        or predecessor.get("exact_stage0_relative_path")
        != "outputs/stage0/v3_m0_solver_addendum_v4.yaml"
        or predecessor.get("required_schema_version")
        != v4_addendum.get("schema_version")
        or predecessor.get("required_status") != v4_addendum.get("status")
    ):
        raise base.ModelError("v5 contract references changed")

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
        or line_search.get("implementation") != "unchanged_v4_byte_for_byte"
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
        or convergence.get("eligible_failure_must_be_all_finite") is not True
        or convergence.get("lambda_must_be_positive_finite") is not True
        or convergence.get("objective_and_full_gradient_must_be_finite")
        is not True
        or convergence.get("prior_successive_relative_change_is_not_overwritten")
        is not True
        or convergence.get("certificate_returns_unchanged_iterate") is not True
        or convergence.get("certificate_adds_accepted_step_or_history_pair")
        is not False
        or _as_float(curvature.get("floor_multiplier"), "curvature floor")
        != CURVATURE_FLOOR_MULTIPLIER
        or curvature.get("skipped_pair_action") != "continue_without_pair"
        or solver.get("fallback_optimizer") != "forbidden"
        or solver.get("other_nonconvergence_or_line_search_failure") != "stop"
    ):
        raise base.ModelError("v5 solver or certificate settings changed")

    if convergence.get("certified_stall_eligible_failure_reasons") != {
        "zoom_iteration_ceiling": "strong-Wolfe zoom iteration ceiling reached",
        "bracket_floating_resolution": (
            "strong-Wolfe zoom bracket reached floating-point resolution"
        ),
    }:
        raise base.ModelError("v5 eligible line-search failures changed")
    if line_search.get("acceptance_requires") != [
        "phi_alpha <= phi_0 + c1 * alpha * directional_derivative_0",
        "abs(directional_derivative_alpha) <= c2 * abs(directional_derivative_0)",
    ]:
        raise base.ModelError("v5 exact strong-Wolfe rules changed")

    certificate = addendum.get("mathematical_certificate", {})
    if (
        certificate.get("conditional_fixed_K_NLL_convex") is not True
        or certificate.get("ridge_covers_every_parameter") is not True
        or int(certificate.get("parameter_dimension", -1))
        != PARAMETER_DIMENSION
        or certificate.get("strong_convexity_modulus") != "lambda"
        or certificate.get("exact_l2_use") != "report_only"
        or certificate.get("conservative_dimension_times_infinity_norm_use")
        != "convergence_gate"
    ):
        raise base.ModelError("v5 strong-convexity statement changed")

    ridge_path = addendum.get("ridge_path", {})
    initialization = ridge_path.get("initialization", {})
    if (
        tuple(float(value) for value in ridge_path.get("ordered_lambdas", ()))
        != DESCENDING_LAMBDAS
        or initialization.get("first_lambda") != "all_zero_coefficients"
        or initialization.get("later_lambdas")
        != "immediately_preceding_converged_coefficients"
        or ridge_path.get("auxiliary_lambdas") != "forbidden"
    ):
        raise base.ModelError("v5 ridge grid or warm-start order changed")

    narrow = addendum.get("narrow_certificate_probe", {})
    full = addendum.get("full_targeted_probe", {})
    if (
        _as_float(narrow.get("ridge_lambda"), "narrow-probe lambda")
        != CERTIFICATE_PROBE_LAMBDA
        or str(narrow.get("outer_held_group")) != PROBE_OUTER_HELD_GROUP
        or str(narrow.get("inner_held_group")) != PROBE_INNER_HELD_GROUP
        or int(narrow.get("expected_training_cells", -1))
        != PROBE_EXPECTED_CELLS
        or int(narrow.get("expected_training_groups", -1))
        != PROBE_EXPECTED_GROUPS
        or tuple(float(value) for value in narrow.get("expected_accepted_steps", ()))
        != CERTIFICATE_PROBE_EXPECTED_STEPS
        or narrow.get("expected_failure_reason") != "zoom_iteration_ceiling"
        or int(narrow.get("expected_failed_search_evaluations", -1))
        != CERTIFICATE_PROBE_FAILURE_EVALUATIONS
        or int(narrow.get("expected_failed_search_zoom_iterations", -1))
        != CERTIFICATE_PROBE_FAILURE_ZOOM_ITERATIONS
        or narrow.get("require_actual_prior_relative_change_to_fail") is not True
        or narrow.get("require_certificate_to_pass") is not True
        or narrow.get("require_unchanged_iterate_and_counters") is not True
        or narrow.get("held_out_scoring") != "forbidden"
        or narrow.get("final_or_unassigned_access") != "forbidden"
        or full.get("runs_only_after_narrow_certificate_probe_passes") is not True
        or str(full.get("outer_held_group")) != PROBE_OUTER_HELD_GROUP
        or str(full.get("inner_held_group")) != PROBE_INNER_HELD_GROUP
        or int(full.get("expected_training_cells", -1)) != PROBE_EXPECTED_CELLS
        or int(full.get("expected_training_groups", -1))
        != PROBE_EXPECTED_GROUPS
        or full.get("full_descending_path_required") is not True
        or int(full.get("deterministic_warm_path_repeats", -1)) != 2
        or full.get("zero_start_comparison_at_every_lambda") is not True
        or int(full.get("warm_path_max_iterations_exclusive", -1))
        != WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
        or full.get("every_accepted_step_must_satisfy_both_Wolfe_conditions")
        is not True
        or full.get("held_out_scoring") != "forbidden"
        or full.get("final_or_unassigned_access") != "forbidden"
    ):
        raise base.ModelError("v5 probe contract changed")

    telemetry = addendum.get("telemetry", {})
    if (
        telemetry.get("scalar_only") is not True
        or telemetry.get("coefficients_prediction_vectors_or_target_contrasts")
        != "forbidden"
    ):
        raise base.ModelError("v5 scalar-only telemetry seal changed")
    return base_contract, addendum


def _finite_or_none(value: float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _stable_l2(values: np.ndarray) -> float:
    return float(math.hypot(*(float(value) for value in values.ravel())))


def _upward_nonnegative(value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        return math.inf
    return math.nextafter(scalar, math.inf)


def _upward_multiply(left: float, right: float) -> float:
    if left < 0.0 or right < 0.0:
        return math.inf
    return _upward_nonnegative(float(left) * float(right))


def _upward_divide(numerator: float, denominator: float) -> float:
    if numerator < 0.0 or not math.isfinite(denominator) or denominator <= 0.0:
        return math.inf
    return _upward_nonnegative(float(numerator) / float(denominator))


def _relative_gap_passes(value: float) -> bool:
    return bool(
        math.isfinite(float(value))
        and float(value) <= RELATIVE_OBJECTIVE_TOLERANCE
    )


def _strong_convexity_certificate(
    *,
    failure_message: str,
    objective: float,
    gradient: np.ndarray,
    ridge_lambda: float,
    parameter_dimension: int,
    prior_relative_change: float,
) -> dict[str, Any]:
    """Return a scalar-only, fail-closed certificate decision."""
    reason = ELIGIBLE_FAILURE_MESSAGES.get(str(failure_message))
    try:
        gradient_array = np.asarray(gradient, dtype=float)
    except (TypeError, ValueError):
        gradient_array = np.asarray([math.nan], dtype=float)

    dimension_matches = bool(
        parameter_dimension > 0
        and gradient_array.ndim == 1
        and gradient_array.size == parameter_dimension
    )
    objective_finite = math.isfinite(float(objective))
    gradient_finite = bool(
        dimension_matches and np.all(np.isfinite(gradient_array))
    )
    lambda_positive_finite = bool(
        math.isfinite(float(ridge_lambda)) and float(ridge_lambda) > 0.0
    )
    exact_l2 = _stable_l2(gradient_array) if gradient_finite else math.inf
    gradient_inf = (
        float(np.max(np.abs(gradient_array))) if gradient_finite else math.inf
    )

    reported_l2_absolute_gap = math.inf
    reported_l2_relative_gap = math.inf
    conservative_absolute_gap = math.inf
    conservative_relative_gap = math.inf
    objective_scale = (
        max(1.0, abs(float(objective))) if objective_finite else math.inf
    )
    if (
        objective_finite
        and gradient_finite
        and lambda_positive_finite
        and math.isfinite(exact_l2)
        and math.isfinite(gradient_inf)
    ):
        l2_square = exact_l2 * exact_l2
        if math.isfinite(l2_square):
            reported_l2_absolute_gap = l2_square / (2.0 * ridge_lambda)
            reported_l2_relative_gap = (
                reported_l2_absolute_gap / objective_scale
            )

        # One binary64 step toward +infinity follows every nonnegative
        # operation.  Division by two is kept separate so no upward-rounded
        # denominator can accidentally make the result smaller.
        gradient_inf_square_up = _upward_multiply(gradient_inf, gradient_inf)
        dimension_times_square_up = _upward_multiply(
            float(parameter_dimension), gradient_inf_square_up
        )
        half_numerator_up = _upward_divide(dimension_times_square_up, 2.0)
        conservative_absolute_gap = _upward_divide(
            half_numerator_up, float(ridge_lambda)
        )
        conservative_relative_gap = _upward_divide(
            conservative_absolute_gap, objective_scale
        )

    failure_eligible = reason is not None
    all_finite_failure_and_state = bool(
        failure_eligible
        and objective_finite
        and gradient_finite
        and math.isfinite(exact_l2)
        and math.isfinite(gradient_inf)
    )
    gradient_pass = bool(
        math.isfinite(gradient_inf) and gradient_inf <= GRADIENT_TOLERANCE
    )
    certified_gap_pass = _relative_gap_passes(conservative_relative_gap)
    certificate_pass = bool(
        failure_eligible
        and all_finite_failure_and_state
        and lambda_positive_finite
        and gradient_pass
        and certified_gap_pass
    )
    return {
        "attempted": failure_eligible,
        "eligible_failure_reason": reason,
        "failure_message": str(failure_message),
        "failure_reason_eligible": failure_eligible,
        "all_prior_line_search_evaluations_finite_by_v4_failure_class": (
            failure_eligible
        ),
        "objective_finite": objective_finite,
        "gradient_finite": gradient_finite,
        "parameter_dimension_matches_full_gradient": dimension_matches,
        "ridge_lambda_positive_finite": lambda_positive_finite,
        "ridge_covers_every_parameter": True,
        "strong_convexity_modulus": (
            float(ridge_lambda) if lambda_positive_finite else None
        ),
        "objective_scale": _finite_or_none(objective_scale),
        "exact_gradient_l2_norm": _finite_or_none(exact_l2),
        "gradient_infinity_norm": _finite_or_none(gradient_inf),
        "reported_l2_absolute_gap_bound": _finite_or_none(
            reported_l2_absolute_gap
        ),
        "reported_l2_relative_gap_bound": _finite_or_none(
            reported_l2_relative_gap
        ),
        "upward_rounded_conservative_absolute_gap_bound": _finite_or_none(
            conservative_absolute_gap
        ),
        "upward_rounded_conservative_relative_gap_bound": _finite_or_none(
            conservative_relative_gap
        ),
        "prior_successive_relative_objective_change": _finite_or_none(
            prior_relative_change
        ),
        "prior_successive_relative_change_pass": bool(
            math.isfinite(float(prior_relative_change))
            and float(prior_relative_change) <= RELATIVE_OBJECTIVE_TOLERANCE
        ),
        "gradient_infinity_tolerance_pass": gradient_pass,
        "certified_relative_gap_tolerance_pass": certified_gap_pass,
        "certificate_pass": certificate_pass,
        "gate_used_exact_l2_bound": False,
        "gate_used_upward_rounded_dimension_times_gradient_inf_bound": True,
    }


def _telemetry_v5(
    *,
    status: str,
    ridge_lambda: float,
    x: np.ndarray,
    groups: Sequence[str],
    vector: np.ndarray,
    objective: float,
    gradient: np.ndarray,
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
    convergence_witness: str,
    line_search_failure: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = v3._telemetry(
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
        expansion_counts=expansion_counts,
        zoom_counts=zoom_counts,
        armijo_slacks=armijo_slacks,
        curvature_ratios=curvature_ratios,
        line_search_failure=line_search_failure,
    )
    gradient_l2 = (
        _stable_l2(np.asarray(gradient, dtype=float))
        if np.all(np.isfinite(gradient))
        else math.inf
    )
    payload.update(
        {
            "solver_version": "v5_strong_convexity_stall_certificate",
            "v4_line_search_unchanged": True,
            "convergence_witness": convergence_witness,
            "exact_gradient_l2_norm": _finite_or_none(gradient_l2),
            "gradient_infinity_tolerance_pass": bool(
                math.isfinite(gradient_norm)
                and gradient_norm <= GRADIENT_TOLERANCE
            ),
            "successive_relative_objective_change_tolerance_pass": bool(
                math.isfinite(relative_change)
                and relative_change <= RELATIVE_OBJECTIVE_TOLERANCE
            ),
            "accepted_step_values": [float(step) for step in step_sizes],
            "coefficient_prediction_or_target_contrast_vectors_emitted": False,
            "strong_convexity_certificate": (
                dict(certificate)
                if certificate is not None
                else {"attempted": False, "certificate_pass": False}
            ),
        }
    )
    return payload


def _fit_lbfgs_v5_core(
    x: np.ndarray,
    y: np.ndarray,
    k_values: np.ndarray,
    groups: Sequence[str],
    target_contrasts: np.ndarray,
    ridge_lambda: float,
    initial: np.ndarray | None,
    *,
    history_size: int,
    maximum_iterations: int,
    gradient_tolerance: float,
    relative_objective_tolerance: float,
    line_search: Callable[
        [
            np.ndarray,
            float,
            np.ndarray,
            np.ndarray,
            Callable[[np.ndarray], tuple[float, np.ndarray]],
        ],
        v3.StrongWolfeResult,
    ],
) -> v3.OptimizerResultV3:
    dimension = x.shape[1] * target_contrasts.shape[1]
    if dimension != PARAMETER_DIMENSION and not (
        x.shape[1] == 1 and target_contrasts.shape[0] == 6
    ):
        raise base.ModelError(f"unexpected v5 optimizer dimension {dimension}")
    if (
        history_size != HISTORY_SIZE
        or maximum_iterations != MAXIMUM_ITERATIONS
        or gradient_tolerance != GRADIENT_TOLERANCE
        or relative_objective_tolerance != RELATIVE_OBJECTIVE_TOLERANCE
    ):
        raise base.ModelError("a caller attempted to override frozen v5 settings")
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
        telemetry = _telemetry_v5(
            status="converged_at_initial_value",
            ridge_lambda=ridge_lambda,
            x=x,
            groups=groups,
            vector=vector,
            objective=value,
            gradient=gradient,
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
            convergence_witness="initial_gradient",
        )
        FIT_TELEMETRY.append(telemetry)
        return v3.OptimizerResultV3(
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
            line = line_search(vector, value, gradient, direction, evaluate)
        except v3.StrongWolfeFailure as error:
            line_search_evaluations += error.evaluations
            vector_before = vector.copy()
            gradient_before = gradient.copy()
            value_before = value
            accepted_before = accepted_steps
            history_before = len(s_history)
            certificate = _strong_convexity_certificate(
                failure_message=str(error),
                objective=value,
                gradient=gradient,
                ridge_lambda=ridge_lambda,
                parameter_dimension=dimension,
                prior_relative_change=relative_change,
            )
            certificate.update(
                {
                    "failed_search_evaluations": int(error.evaluations),
                    "failed_search_expansions": int(error.expansions),
                    "failed_search_zoom_iterations": int(error.zoom_iterations),
                    "accepted_steps_before_certificate": int(accepted_before),
                    "accepted_steps_after_certificate": int(accepted_steps),
                    "history_pairs_before_certificate": int(history_before),
                    "history_pairs_after_certificate": int(len(s_history)),
                    "iterate_array_unchanged": bool(
                        np.array_equal(vector, vector_before)
                    ),
                    "objective_unchanged": bool(value == value_before),
                    "gradient_array_unchanged": bool(
                        np.array_equal(gradient, gradient_before)
                    ),
                    "coefficient_l2_norm_before": float(
                        np.linalg.norm(vector_before)
                    ),
                    "coefficient_l2_norm_after": float(np.linalg.norm(vector)),
                    "no_pseudo_step": bool(
                        accepted_steps == accepted_before
                        and len(s_history) == history_before
                        and np.array_equal(vector, vector_before)
                        and value == value_before
                        and np.array_equal(gradient, gradient_before)
                    ),
                }
            )
            failure_payload = {
                "message": str(error),
                "evaluations": error.evaluations,
                "expansions": error.expansions,
                "zoom_iterations": error.zoom_iterations,
            }
            if certificate["certificate_pass"]:
                if not certificate["no_pseudo_step"]:
                    raise base.ModelError(
                        "v5 certificate attempted to mutate the current iterate"
                    )
                telemetry = _telemetry_v5(
                    status=(
                        "converged_after_line_search_precision_failure_by_"
                        "strong_convexity_certificate"
                    ),
                    ridge_lambda=ridge_lambda,
                    x=x,
                    groups=groups,
                    vector=vector,
                    objective=value,
                    gradient=gradient,
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
                    convergence_witness="strong_convexity_gap_after_v4_zoom_stall",
                    line_search_failure=failure_payload,
                    certificate=certificate,
                )
                FIT_TELEMETRY.append(telemetry)
                return v3.OptimizerResultV3(
                    coefficients=vector.reshape(
                        x.shape[1], target_contrasts.shape[1]
                    ),
                    objective=value,
                    gradient_infinity_norm=gradient_norm,
                    iterations=iteration - 1,
                    accepted_steps=accepted_steps,
                    relative_objective_change=relative_change,
                    solver_telemetry=telemetry,
                )

            telemetry = _telemetry_v5(
                status="line_search_failure",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient=gradient,
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
                convergence_witness="none",
                line_search_failure=failure_payload,
                certificate=certificate,
            )
            FIT_TELEMETRY.append(telemetry)
            raise base.ModelError(
                "v5 unchanged strong-Wolfe line search failed without a valid "
                "strong-convexity certificate; telemetry="
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
            telemetry = _telemetry_v5(
                status="converged",
                ridge_lambda=ridge_lambda,
                x=x,
                groups=groups,
                vector=vector,
                objective=value,
                gradient=gradient,
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
                convergence_witness="successive_relative_objective_change",
            )
            FIT_TELEMETRY.append(telemetry)
            return v3.OptimizerResultV3(
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

    telemetry = _telemetry_v5(
        status="iteration_cap_exhausted",
        ridge_lambda=ridge_lambda,
        x=x,
        groups=groups,
        vector=vector,
        objective=value,
        gradient=gradient,
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
        convergence_witness="none",
    )
    FIT_TELEMETRY.append(telemetry)
    raise base.ModelError(
        "v5 deterministic L-BFGS did not meet an unchanged convergence witness "
        f"in {maximum_iterations} iterations; telemetry="
        + json.dumps(telemetry, sort_keys=True)
    )


def fit_lbfgs_v5(
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
) -> v3.OptimizerResultV3:
    return _fit_lbfgs_v5_core(
        x,
        y,
        k_values,
        groups,
        target_contrasts,
        ridge_lambda,
        initial,
        history_size=history_size,
        maximum_iterations=maximum_iterations,
        gradient_tolerance=gradient_tolerance,
        relative_objective_tolerance=relative_objective_tolerance,
        line_search=V4_STRONG_WOLFE_SEARCH,
    )


# Every unchanged v1 fitting route and the retained v2 probe/stress scaffolds
# resolve to V5, while V4_FIT_LBFGS_ORIGINAL remains available for parity tests.
base.fit_lbfgs = fit_lbfgs_v5
v2.fit_lbfgs_v2 = fit_lbfgs_v5
v2.FIT_TELEMETRY = FIT_TELEMETRY


def _certificate_arithmetic_self_test() -> dict[str, Any]:
    trace_gradient = np.zeros(PARAMETER_DIMENSION, dtype=float)
    trace_gradient[0] = 2.919646807653109e-8
    trace = _strong_convexity_certificate(
        failure_message="strong-Wolfe zoom iteration ceiling reached",
        objective=6.723861954254116,
        gradient=trace_gradient,
        ridge_lambda=1000.0,
        parameter_dimension=PARAMETER_DIMENSION,
        prior_relative_change=1.2808392185936424e-7,
    )
    if (
        not trace["certificate_pass"]
        or trace["prior_successive_relative_change_pass"]
        or trace["upward_rounded_conservative_relative_gap_bound"]
        > 1.617e-17
    ):
        raise AssertionError("traced-state certificate regression failed")

    exact_gradient_boundary = _strong_convexity_certificate(
        failure_message="strong-Wolfe zoom iteration ceiling reached",
        objective=1.0,
        gradient=np.asarray([GRADIENT_TOLERANCE]),
        ridge_lambda=1.0,
        parameter_dimension=1,
        prior_relative_change=1.0,
    )
    above_gradient_boundary = _strong_convexity_certificate(
        failure_message="strong-Wolfe zoom iteration ceiling reached",
        objective=1.0,
        gradient=np.asarray(
            [math.nextafter(GRADIENT_TOLERANCE, math.inf)]
        ),
        ridge_lambda=1.0,
        parameter_dimension=1,
        prior_relative_change=1.0,
    )
    weak_ridge = _strong_convexity_certificate(
        failure_message="strong-Wolfe zoom iteration ceiling reached",
        objective=1.0,
        gradient=np.asarray([GRADIENT_TOLERANCE]),
        ridge_lambda=1.0e-8,
        parameter_dimension=1,
        prior_relative_change=1.0,
    )
    if (
        not exact_gradient_boundary["gradient_infinity_tolerance_pass"]
        or above_gradient_boundary["gradient_infinity_tolerance_pass"]
        or weak_ridge["certified_relative_gap_tolerance_pass"]
    ):
        raise AssertionError("certificate tolerance boundary regression failed")
    if not _relative_gap_passes(RELATIVE_OBJECTIVE_TOLERANCE) or _relative_gap_passes(
        math.nextafter(RELATIVE_OBJECTIVE_TOLERANCE, math.inf)
    ):
        raise AssertionError("relative-gap exact comparator boundary failed")

    negative_controls: list[dict[str, Any]] = []
    for label, ridge_lambda in (
        ("zero_lambda", 0.0),
        ("negative_lambda", -1.0),
        ("nan_lambda", math.nan),
        ("infinite_lambda", math.inf),
    ):
        decision = _strong_convexity_certificate(
            failure_message="strong-Wolfe zoom iteration ceiling reached",
            objective=1.0,
            gradient=np.asarray([1.0e-9]),
            ridge_lambda=ridge_lambda,
            parameter_dimension=1,
            prior_relative_change=1.0,
        )
        if decision["certificate_pass"]:
            raise AssertionError(f"certificate accepted {label}")
        negative_controls.append({"case": label, "status": "pass"})

    for label, objective, gradient in (
        ("nan_objective", math.nan, np.asarray([1.0e-9])),
        ("infinite_objective", math.inf, np.asarray([1.0e-9])),
        ("nan_gradient", 1.0, np.asarray([math.nan])),
        ("infinite_gradient", 1.0, np.asarray([math.inf])),
    ):
        decision = _strong_convexity_certificate(
            failure_message="strong-Wolfe zoom iteration ceiling reached",
            objective=objective,
            gradient=gradient,
            ridge_lambda=1.0,
            parameter_dimension=1,
            prior_relative_change=1.0,
        )
        if decision["certificate_pass"]:
            raise AssertionError(f"certificate accepted {label}")
        negative_controls.append({"case": label, "status": "pass"})

    for message in (
        "strong-Wolfe evaluation ceiling reached",
        "strong-Wolfe search received a nonfinite initial state",
        "strong-Wolfe search requires a descent direction",
        "internal strong-Wolfe acceptance violation",
        "strong-Wolfe maximum step reached without a bracket",
    ):
        decision = _strong_convexity_certificate(
            failure_message=message,
            objective=math.nextafter(1.0, math.inf),
            gradient=np.asarray([0.0]),
            ridge_lambda=1.0,
            parameter_dimension=1,
            prior_relative_change=0.0,
        )
        if decision["certificate_pass"]:
            raise AssertionError("certificate accepted a noneligible failure")
        negative_controls.append({"case": message, "status": "pass"})

    return {
        "status": "pass",
        "traced_state": trace,
        "gradient_exact_boundary_pass": True,
        "gradient_just_over_boundary_rejected": True,
        "relative_gap_exact_boundary_comparator_pass": True,
        "relative_gap_just_over_boundary_comparator_rejected": True,
        "weak_ridge_gap_rejected": True,
        "negative_controls": negative_controls,
    }


def _known_quadratic_bound_self_test() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for diagonal, point in (
        (np.asarray([3.0]), np.asarray([0.25])),
        (np.asarray([2.0, 7.0]), np.asarray([0.3, -0.2])),
        (np.asarray([0.5, 1.5, 4.0]), np.asarray([0.1, -0.4, 0.2])),
    ):
        ridge_lambda = float(np.min(diagonal))
        gradient = diagonal * point
        objective = 0.5 * float(np.dot(diagonal * point, point))
        decision = _strong_convexity_certificate(
            failure_message="strong-Wolfe zoom iteration ceiling reached",
            objective=objective,
            gradient=gradient,
            ridge_lambda=ridge_lambda,
            parameter_dimension=point.size,
            prior_relative_change=1.0,
        )
        l2_bound = float(decision["reported_l2_absolute_gap_bound"])
        conservative = float(
            decision["upward_rounded_conservative_absolute_gap_bound"]
        )
        if l2_bound + 1.0e-15 < objective or conservative < objective:
            raise AssertionError("strong-convexity bound understated a known gap")
        cases.append(
            {
                "dimension": int(point.size),
                "modulus": ridge_lambda,
                "actual_gap": objective,
                "reported_l2_gap_bound": l2_bound,
                "conservative_gap_bound": conservative,
            }
        )
    return {"status": "pass", "known_optimum_quadratics": cases}


def _rounded_evaluator_stall_self_test() -> dict[str, Any]:
    baseline = 1.0e16
    baseline_ulp = math.ulp(baseline)
    ridge_lambda = 1000.0
    initial_gradient_scalar = 5.0e-8
    start = np.asarray([0.0])
    gradient = np.asarray([initial_gradient_scalar])
    direction = -gradient
    start_copy = start.copy()

    def rounded_evaluator(point: np.ndarray) -> tuple[float, np.ndarray]:
        alpha = float(point[0] / direction[0])
        # This deterministic scalar evaluator models a few-ULP upward summation
        # perturbation exactly where the underlying ridge quadratic has valid
        # curvature, and a rounded-flat baseline once curvature has failed.
        represented_value = (
            baseline + 4.0 * baseline_ulp if alpha >= 1.0e-4 else baseline
        )
        exact_gradient = np.asarray(
            [initial_gradient_scalar + ridge_lambda * float(point[0])]
        )
        return represented_value, exact_gradient

    try:
        V4_STRONG_WOLFE_SEARCH(
            start,
            baseline,
            gradient,
            direction,
            rounded_evaluator,
        )
    except v3.StrongWolfeFailure as error:
        if str(error) not in ELIGIBLE_FAILURE_MESSAGES:
            raise AssertionError("rounded evaluator failed for an ineligible reason")
        failure = error
    else:
        raise AssertionError("rounded evaluator unexpectedly found a Wolfe step")

    decision = _strong_convexity_certificate(
        failure_message=str(failure),
        objective=baseline,
        gradient=gradient,
        ridge_lambda=ridge_lambda,
        parameter_dimension=1,
        prior_relative_change=1.0,
    )
    if not decision["certificate_pass"] or not np.array_equal(start, start_copy):
        raise AssertionError("rounded evaluator stall did not certify unchanged")
    return {
        "status": "pass",
        "failure_reason": decision["eligible_failure_reason"],
        "evaluations": failure.evaluations,
        "zoom_iterations": failure.zoom_iterations,
        "current_iterate_unchanged": True,
        "no_Wolfe_step_fabricated": True,
        "certificate": decision,
    }


def _full_ridge_coverage_self_test() -> dict[str, Any]:
    contrasts = base.helmert(16)
    x = np.linspace(-0.8, 0.9, 68, dtype=float).reshape(4, 17)
    y = np.zeros((4, 16), dtype=float)
    y[0, [0, 3]] = 1.0
    y[1, [2, 5, 8]] = 1.0
    y[2, [1, 7, 10, 14]] = 1.0
    y[3, [4, 9, 15]] = 1.0
    k_values = np.asarray([2, 3, 4, 3], dtype=int)
    groups = ["g0", "g1", "g2", "g3"]
    vector = np.linspace(-0.31, 0.29, PARAMETER_DIMENSION, dtype=float)
    ridge_lambda = 2.75
    value_zero, gradient_zero = base.objective_and_gradient(
        vector, x, y, k_values, groups, contrasts, 0.0
    )
    value_ridge, gradient_ridge = base.objective_and_gradient(
        vector, x, y, k_values, groups, contrasts, ridge_lambda
    )
    expected_value_delta = 0.5 * ridge_lambda * float(np.dot(vector, vector))
    observed_gradient_delta = gradient_ridge - gradient_zero
    expected_gradient_delta = ridge_lambda * vector
    value_error = abs((value_ridge - value_zero) - expected_value_delta)
    gradient_error = float(
        np.max(np.abs(observed_gradient_delta - expected_gradient_delta))
    )
    if value_error > 2.0e-13 or gradient_error > 2.0e-14:
        raise AssertionError("full-parameter ridge coverage regression failed")
    if np.count_nonzero(observed_gradient_delta) != PARAMETER_DIMENSION:
        raise AssertionError("ridge did not cover every M0 parameter")
    return {
        "status": "pass",
        "parameter_dimension": PARAMETER_DIMENSION,
        "ridge_nonzero_gradient_coordinates": int(
            np.count_nonzero(observed_gradient_delta)
        ),
        "objective_ridge_identity_absolute_error": value_error,
        "gradient_ridge_identity_max_absolute_error": gradient_error,
    }


def _ordinary_v4_parity_self_test() -> dict[str, Any]:
    contrasts = base.helmert(6)
    x = np.ones((24, 1), dtype=float)
    k_values = np.full(24, 3, dtype=int)
    groups = [f"g{index // 6}" for index in range(24)]
    planted = np.asarray([[0.31, -0.23, 0.17, -0.11, 0.07]])
    eta = (x @ planted) @ contrasts.T
    _, y = base.batch_fixed_k_logz_marginals(eta, k_values)

    v3.FIT_TELEMETRY.clear()
    v4_result = V4_FIT_LBFGS_ORIGINAL(
        x, y, k_values, groups, contrasts, 3.0
    )
    FIT_TELEMETRY.clear()
    v5_result = fit_lbfgs_v5(x, y, k_values, groups, contrasts, 3.0)
    shared_telemetry_keys = (
        "accepted_step_summary",
        "initial_inverse_hessian_scale_summary",
        "line_search_evaluations",
        "curvature_updates",
        "curvature_skips",
        "line_search_expansion_count_summary",
        "line_search_zoom_iteration_summary",
        "accepted_armijo_slack_summary",
        "accepted_curvature_ratio_summary",
        "strong_wolfe_all_accepted_steps_pass",
    )
    if (
        v4_result.accepted_steps <= 0
        or not np.array_equal(v4_result.coefficients, v5_result.coefficients)
        or v4_result.objective != v5_result.objective
        or v4_result.gradient_infinity_norm != v5_result.gradient_infinity_norm
        or v4_result.iterations != v5_result.iterations
        or v4_result.accepted_steps != v5_result.accepted_steps
        or v4_result.relative_objective_change
        != v5_result.relative_objective_change
        or any(
            v4_result.solver_telemetry[key]
            != v5_result.solver_telemetry[key]
            for key in shared_telemetry_keys
        )
        or v5_result.solver_telemetry["convergence_witness"]
        != "successive_relative_objective_change"
    ):
        raise AssertionError("ordinary V4/V5 path lost bitwise parity")
    return {
        "status": "pass",
        "accepted_steps": v5_result.accepted_steps,
        "iterations": v5_result.iterations,
        "objective_hex": float(v5_result.objective).hex(),
        "coefficient_arrays_bitwise_equal": True,
        "ordinary_result_scalars_bitwise_equal": True,
        "shared_Wolfe_telemetry_equal": True,
    }


def _synthetic_certificate_termination_self_test() -> dict[str, Any]:
    """Exercise the actual V5 failure branch after two real Wolfe steps."""
    contrasts = base.helmert(6)
    x = np.ones((24, 1), dtype=float)
    k_values = np.full(24, 3, dtype=int)
    groups = [f"g{index // 6}" for index in range(24)]
    planted = np.asarray([[0.31, -0.23, 0.17, -0.11, 0.07]])
    eta = (x @ planted) @ contrasts.T
    _, y = base.batch_fixed_k_logz_marginals(eta, k_values)
    search_calls = 0
    failed_iterate: np.ndarray | None = None

    def two_steps_then_precision_stall(
        vector: np.ndarray,
        value: float,
        gradient: np.ndarray,
        direction: np.ndarray,
        evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
    ) -> v3.StrongWolfeResult:
        nonlocal search_calls, failed_iterate
        search_calls += 1
        if search_calls == 3:
            failed_iterate = vector.copy()
            raise v3.StrongWolfeFailure(
                "strong-Wolfe zoom iteration ceiling reached",
                evaluations=CERTIFICATE_PROBE_FAILURE_EVALUATIONS,
                expansions=0,
                zoom_iterations=CERTIFICATE_PROBE_FAILURE_ZOOM_ITERATIONS,
            )
        return V4_STRONG_WOLFE_SEARCH(
            vector, value, gradient, direction, evaluate
        )

    FIT_TELEMETRY.clear()
    result = _fit_lbfgs_v5_core(
        x,
        y,
        k_values,
        groups,
        contrasts,
        CERTIFICATE_PROBE_LAMBDA,
        None,
        history_size=HISTORY_SIZE,
        maximum_iterations=MAXIMUM_ITERATIONS,
        gradient_tolerance=GRADIENT_TOLERANCE,
        relative_objective_tolerance=RELATIVE_OBJECTIVE_TOLERANCE,
        line_search=two_steps_then_precision_stall,
    )
    telemetry = result.solver_telemetry
    certificate = telemetry["strong_convexity_certificate"]
    if (
        failed_iterate is None
        or result.iterations != 2
        or result.accepted_steps != 2
        or tuple(telemetry["accepted_step_values"])
        != CERTIFICATE_PROBE_EXPECTED_STEPS
        or result.relative_objective_change <= RELATIVE_OBJECTIVE_TOLERANCE
        or telemetry["successive_relative_objective_change_tolerance_pass"]
        is not False
        or certificate["certificate_pass"] is not True
        or certificate["no_pseudo_step"] is not True
        or certificate["accepted_steps_before_certificate"] != 2
        or certificate["accepted_steps_after_certificate"] != 2
        or certificate["history_pairs_before_certificate"]
        != certificate["history_pairs_after_certificate"]
        or not np.array_equal(result.coefficients.ravel(), failed_iterate)
        or telemetry["strong_wolfe_all_accepted_steps_pass"] is not True
    ):
        raise AssertionError("synthetic V5 certificate termination regression failed")
    return {
        "status": "pass",
        "real_v4_Wolfe_steps_before_injected_stall": list(
            telemetry["accepted_step_values"]
        ),
        "iterations": result.iterations,
        "accepted_steps": result.accepted_steps,
        "actual_prior_relative_objective_change": (
            result.relative_objective_change
        ),
        "actual_prior_relative_change_still_failed": True,
        "returned_iterate_equals_prefailure_iterate": True,
        "accepted_step_and_history_counts_unchanged": True,
        "no_pseudo_step": True,
        "certificate": certificate,
    }


def run_certificate_unit_tests() -> dict[str, Any]:
    """Fast, data-free tests suitable for a login-node syntax/unit gate."""
    result = {
        "status": "pass",
        "certificate_arithmetic": _certificate_arithmetic_self_test(),
        "known_quadratic_bounds": _known_quadratic_bound_self_test(),
        "rounded_evaluator_stall": _rounded_evaluator_stall_self_test(),
        "full_ridge_coverage": _full_ridge_coverage_self_test(),
        "ordinary_v4_parity": _ordinary_v4_parity_self_test(),
        "synthetic_certificate_termination": (
            _synthetic_certificate_termination_self_test()
        ),
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }
    FIT_TELEMETRY.clear()
    v3.FIT_TELEMETRY.clear()
    return result


def run_self_test(base_contract_path: Path, addendum_path: Path) -> dict[str, Any]:
    validate_solver_addendum(base_contract_path, addendum_path)
    certificate_tests = run_certificate_unit_tests()
    retained_v4_line_search = v4._line_search_self_test()
    base_result = base.run_self_test(base_contract_path)
    stress = v2._synthetic_ill_conditioned_stress()
    for label in ("warm_fit", "stress_fit"):
        if stress[label].get("strong_wolfe_all_accepted_steps_pass") is not True:
            raise AssertionError(f"full-dimension stress {label} violated Wolfe")
    FIT_TELEMETRY.clear()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v5_synthetic_self_test",
        "v5_certificate_tests": certificate_tests,
        "retained_v4_line_search_checks": retained_v4_line_search,
        "base_v1_self_test": base_result,
        "full_dimension_ill_conditioned_stress": stress,
        "uses_real_data": False,
        "final_or_unassigned_opened": False,
    }


def _prepare_unique_output(path: Path, scratch: Path, *, mode: str) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real v5 solver work must run in a Slurm job")
    episode_root = v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    if not path.is_absolute():
        raise base.ModelError("output root must be absolute")
    absolute = Path(os.path.abspath(path))
    expected_names = {
        "certificate_probe": f"v3_m0_solver_certificate_probe_v5_job_{job_id}",
        "full_probe": f"v3_m0_solver_probe_v5_job_{job_id}",
        "execute": PRODUCTION_OUTPUT_BASENAME,
    }
    expected_name = expected_names[mode]
    if absolute.parent != episode_root or absolute.name != expected_name:
        raise base.ModelError(
            f"output must be the exact EP10 Scratch child {expected_name!r}"
        )
    if os.path.lexists(absolute):
        raise base.ModelError("v5 output root must be absent")
    if mode != "execute":
        absolute.mkdir(parents=False)
        if absolute != absolute.resolve(strict=True) or absolute.is_symlink():
            raise base.ModelError("v5 probe output root failed its path seal")
    return absolute


def _load_exact_probe_training(
    target_root: Path, interface_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    records = base.load_real_records(target_root, interface_root)
    training = [
        record
        for record in records
        if record.group not in (PROBE_OUTER_HELD_GROUP, PROBE_INNER_HELD_GROUP)
    ]
    training_groups = sorted(set(record.group for record in training))
    if (
        len(training) != PROBE_EXPECTED_CELLS
        or len(training_groups) != PROBE_EXPECTED_GROUPS
        or PROBE_OUTER_HELD_GROUP in training_groups
        or PROBE_INNER_HELD_GROUP in training_groups
    ):
        raise base.ModelError("targeted v5 probe training split changed")
    transform = base.fit_design_transform(training)
    x = base.build_design(training, transform)
    y, k_values, groups = base.outcome_matrix(training)
    return x, y, k_values, groups


def run_certificate_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(
        args.output_root, scratch, mode="certificate_probe"
    )
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v5_narrow_certificate_probe",
        "ridge_lambda": CERTIFICATE_PROBE_LAMBDA,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_certificate_probe_v5.json"
    try:
        x, y, k_values, groups = _load_exact_probe_training(
            target_root, interface_root
        )
        FIT_TELEMETRY.clear()
        result = fit_lbfgs_v5(
            x,
            y,
            k_values,
            groups,
            base.helmert(16),
            CERTIFICATE_PROBE_LAMBDA,
        )
        telemetry = result.solver_telemetry
        certificate = telemetry.get("strong_convexity_certificate", {})
        failure = telemetry.get("line_search_failure", {})
        if (
            telemetry.get("status")
            != "converged_after_line_search_precision_failure_by_strong_convexity_certificate"
            or tuple(telemetry.get("accepted_step_values", ()))
            != CERTIFICATE_PROBE_EXPECTED_STEPS
            or result.iterations != 2
            or result.accepted_steps != 2
            or result.relative_objective_change
            <= RELATIVE_OBJECTIVE_TOLERANCE
            or telemetry.get(
                "successive_relative_objective_change_tolerance_pass"
            )
            is not False
            or failure.get("message")
            != "strong-Wolfe zoom iteration ceiling reached"
            or int(failure.get("evaluations", -1))
            != CERTIFICATE_PROBE_FAILURE_EVALUATIONS
            or int(failure.get("zoom_iterations", -1))
            != CERTIFICATE_PROBE_FAILURE_ZOOM_ITERATIONS
            or certificate.get("eligible_failure_reason")
            != "zoom_iteration_ceiling"
            or certificate.get("certificate_pass") is not True
            or certificate.get("gradient_infinity_tolerance_pass") is not True
            or certificate.get("certified_relative_gap_tolerance_pass") is not True
            or certificate.get("no_pseudo_step") is not True
            or certificate.get("iterate_array_unchanged") is not True
            or certificate.get("objective_unchanged") is not True
            or certificate.get("gradient_array_unchanged") is not True
            or certificate.get("accepted_steps_before_certificate")
            != certificate.get("accepted_steps_after_certificate")
            or certificate.get("history_pairs_before_certificate")
            != certificate.get("history_pairs_after_certificate")
            or certificate.get("exact_gradient_l2_norm") is None
            or telemetry.get("strong_wolfe_all_accepted_steps_pass") is not True
        ):
            raise base.ModelError("narrow v5 certificate probe signature changed")
        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_groups": int(len(set(groups))),
                "accepted_steps": result.accepted_steps,
                "iterations": result.iterations,
                "actual_prior_relative_objective_change": (
                    result.relative_objective_change
                ),
                "actual_prior_relative_change_still_failed": True,
                "certificate_telemetry": telemetry,
                "unchanged_iterate_and_counters_pass": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
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


def _result_has_valid_v5_witness(result: v3.OptimizerResultV3) -> bool:
    telemetry = result.solver_telemetry
    literal = bool(
        result.gradient_infinity_norm <= GRADIENT_TOLERANCE
        and result.relative_objective_change <= RELATIVE_OBJECTIVE_TOLERANCE
    )
    certificate = telemetry.get("strong_convexity_certificate", {})
    certified = bool(
        result.gradient_infinity_norm <= GRADIENT_TOLERANCE
        and certificate.get("certificate_pass") is True
        and certificate.get("no_pseudo_step") is True
    )
    return literal or certified


def run_targeted_probe(args: argparse.Namespace) -> dict[str, Any]:
    validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _prepare_unique_output(
        args.output_root, scratch, mode="full_probe"
    )
    FIT_TELEMETRY.clear()
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v5_full_targeted_numerical_probe",
        "solver_addendum": args.solver_addendum.name,
        "requires_prior_narrow_certificate_probe_pass": True,
        "outer_held_group": PROBE_OUTER_HELD_GROUP,
        "inner_held_group": PROBE_INNER_HELD_GROUP,
        "held_out_scores_emitted": False,
        "coefficient_prediction_or_target_contrast_vectors_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    output_path = output_root / "v3_m0_solver_probe_v5.json"
    try:
        certificate_tests = run_certificate_unit_tests()
        retained_line_search = v4._line_search_self_test()
        stress = v2._synthetic_ill_conditioned_stress()
        FIT_TELEMETRY.clear()
        x, y, k_values, groups = _load_exact_probe_training(
            target_root, interface_root
        )

        warm_first = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=True
        )
        first_telemetry = [
            warm_first[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        warm_second = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=True
        )
        second_telemetry = [
            warm_second[value][0].solver_telemetry
            for value in DESCENDING_LAMBDAS
        ]
        zero_start = v2._fit_probe_path(
            x, y, k_values, groups, warm_start=False
        )
        zero_telemetry = [
            zero_start[value][0].solver_telemetry for value in DESCENDING_LAMBDAS
        ]
        repeat_checks, zero_checks = v2._compare_probe_paths(
            warm_first, warm_second, zero_start
        )
        all_results = [
            result
            for path in (warm_first, warm_second, zero_start)
            for result, _ in path.values()
        ]
        warm_path_worst_iterations = max(
            result.iterations
            for path in (warm_first, warm_second)
            for result, _ in path.values()
        )
        if warm_path_worst_iterations >= WARM_PATH_MAX_ITERATIONS_EXCLUSIVE:
            raise base.ModelError(
                "v5 probe lacks frozen warm-path iteration headroom: "
                f"worst={warm_path_worst_iterations}, required <"
                f"{WARM_PATH_MAX_ITERATIONS_EXCLUSIVE}"
            )
        if not all(
            result.solver_telemetry.get("strong_wolfe_all_accepted_steps_pass")
            is True
            for result in all_results
        ):
            raise base.ModelError("a v5 probe fit accepted a non-Wolfe step")
        if not all(_result_has_valid_v5_witness(result) for result in all_results):
            raise base.ModelError("a v5 probe fit lacks a valid convergence witness")
        if not all(item["pass"] for item in repeat_checks):
            raise base.ModelError("v5 deterministic repeat check failed")
        if not all(item["pass"] for item in zero_checks):
            raise base.ModelError("v5 warm-start versus zero-start check failed")

        certificate_fit_count = sum(
            result.solver_telemetry.get("strong_convexity_certificate", {}).get(
                "certificate_pass"
            )
            is True
            for result in all_results
        )
        payload.update(
            {
                "status": "pass",
                "training_cells": int(x.shape[0]),
                "training_groups": int(len(set(groups))),
                "descending_lambda_order": list(DESCENDING_LAMBDAS),
                "v5_certificate_unit_checks": certificate_tests,
                "retained_v4_line_search_checks": retained_line_search,
                "synthetic_full_dimension_stress": stress,
                "warm_path_repeat_1": first_telemetry,
                "warm_path_repeat_2": second_telemetry,
                "zero_start_path": zero_telemetry,
                "deterministic_repeat_checks": repeat_checks,
                "warm_start_vs_zero_start_checks": zero_checks,
                "warm_path_worst_iterations": warm_path_worst_iterations,
                "warm_path_max_iterations_exclusive": (
                    WARM_PATH_MAX_ITERATIONS_EXCLUSIVE
                ),
                "warm_path_iteration_headroom_pass": True,
                "fits_using_strong_convexity_certificate": certificate_fit_count,
                "all_fits_met_gradient_and_literal_or_certified_objective_accuracy": True,
                "all_accepted_steps_satisfied_exact_strong_Wolfe": True,
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
    output_root = _prepare_unique_output(args.output_root, scratch, mode="execute")
    args.target_root = target_root
    args.interface_root = interface_root
    args.output_root = output_root
    FIT_TELEMETRY.clear()
    summary = base.run_real(args)
    summary["solver_addendum_v5"] = {
        "schema_version": "episode10_v3_m0_solver_addendum_v5",
        "history_size": HISTORY_SIZE,
        "maximum_iterations": MAXIMUM_ITERATIONS,
        "v4_line_search_unchanged": True,
        "convergence_tolerances_changed": False,
        "certified_stall_witness_added": True,
    }
    base._atomic_json(output_root / "v3_m0_fit_summary.json", summary)
    base._atomic_json(
        output_root / "v3_m0_solver_telemetry_v5.json",
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
    mode.add_argument("--unit-test", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--certificate-probe", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument("--solver-addendum", type=Path, default=V5_ADDENDUM_PATH)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--temperature-diagnostic", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.unit_test or args.self_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("test modes do not accept real input or output paths")
        if args.temperature_diagnostic:
            parser.error("--temperature-diagnostic is unavailable in test modes")
        try:
            if args.unit_test:
                validate_solver_addendum(args.contract, args.solver_addendum)
                result = run_certificate_unit_tests()
            else:
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
    if (args.probe or args.certificate_probe) and args.temperature_diagnostic:
        parser.error("numerical probes cannot run the temperature diagnostic")
    try:
        if args.certificate_probe:
            result = run_certificate_probe(args)
        elif args.probe:
            result = run_targeted_probe(args)
        else:
            result = run_execute(args)
        if args.certificate_probe or args.probe:
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
    except base.BoundaryReview as error:
        print(f"CONTRACT REVIEW REQUIRED: {error}", file=sys.stderr)
        return 3
    except (base.ModelError, AssertionError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
