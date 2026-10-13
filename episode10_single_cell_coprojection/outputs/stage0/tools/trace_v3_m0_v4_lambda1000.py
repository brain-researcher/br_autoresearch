#!/usr/bin/env python3
"""Scalar-only diagnostic trace of the frozen EP10 M0 v4 lambda-1000 fit."""

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
    raise ImportError("trace-v1 script must be an unsymlinked regular file")
SCRIPT_DIR = SCRIPT_PATH.parent
STAGE0_DIR = SCRIPT_DIR.parent
V4_ENGINE_PATH = SCRIPT_DIR / "fit_v3_m0_v4.py"
if (
    not V4_ENGINE_PATH.is_file()
    or V4_ENGINE_PATH.is_symlink()
    or V4_ENGINE_PATH != V4_ENGINE_PATH.resolve(strict=True)
):
    raise ImportError("v4 engine must be the exact unsymlinked stage0 file")

_V4_SPEC = importlib.util.spec_from_file_location(
    "ep10_v3_m0_solver_v4_for_trace_v1", V4_ENGINE_PATH
)
if _V4_SPEC is None or _V4_SPEC.loader is None:
    raise ImportError(f"cannot import v4 engine: {V4_ENGINE_PATH}")
v4 = importlib.util.module_from_spec(_V4_SPEC)
sys.modules[_V4_SPEC.name] = v4
_V4_SPEC.loader.exec_module(v4)
base = v4.base


BASE_CONTRACT_PATH = STAGE0_DIR / "v3_m0_model_contract.yaml"
V4_ADDENDUM_PATH = STAGE0_DIR / "v3_m0_solver_addendum_v4.yaml"
RIDGE_LAMBDA = 1000.0
OUTER_HELD_GROUP = "212074"
INNER_HELD_GROUP = "220187"
EXPECTED_TRAINING_CELLS = 153
EXPECTED_TRAINING_GROUPS = 10
EXPECTED_ACCEPTED_STEPS = (math.ldexp(1.0, -10), 1.0)
EXPECTED_FAILURE_EVALUATIONS = 61
EXPECTED_FAILURE_ZOOM_ITERATIONS = 60
OUTPUT_BASENAME_PREFIX = "v3_m0_solver_trace_v1_job_"
OUTPUT_JSON_BASENAME = "v3_m0_solver_trace_v1.json"
RUNTIME_LOG_BASENAME = "runtime.log"


class TraceDrift(base.ModelError):
    """Raised when the frozen trace no longer has its expected signature."""


@dataclass(frozen=True)
class CapturedTrial:
    candidate: np.ndarray
    value: float
    gradient: np.ndarray


@dataclass(frozen=True)
class TracedSearch:
    result: Any | None
    failure: Any | None
    events: list[dict[str, Any]]
    summary: dict[str, Any]


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
        raise base.ModelError(f"{label} is not the exact unsymlinked stage0 file")
    return absolute


def _finite_or_none(value: float) -> float | None:
    scalar = float(value)
    return scalar if math.isfinite(scalar) else None


def _float_hex(value: float) -> str:
    return float(value).hex()


def _sign(value: float) -> int:
    return int(value > 0.0) - int(value < 0.0)


def _ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0 or not math.isfinite(denominator):
        return None
    return _finite_or_none(numerator / denominator)


def _bracket(low: float, high: float) -> dict[str, Any]:
    orientation_value = high - low
    return {
        "low": float(low),
        "low_hex": _float_hex(low),
        "high": float(high),
        "high_hex": _float_hex(high),
        "orientation_signed": float(orientation_value),
        "orientation": (
            "ascending"
            if orientation_value > 0.0
            else "descending"
            if orientation_value < 0.0
            else "degenerate"
        ),
        "width": abs(float(orientation_value)),
    }


def _trial_event(
    *,
    context: Mapping[str, Any],
    phase: str,
    phase_iteration: int,
    trial_index: int,
    evaluations_before: int,
    evaluations_after: int,
    expansions_before: int,
    expansions_after: int,
    zoom_before: int,
    zoom_after: int,
    alpha: float,
    phi0: float,
    phi: float,
    phi_lo: float,
    armijo_rhs: float,
    dphi0: float,
    dphi: float,
    candidate: np.ndarray,
    current: np.ndarray,
    previous_trial: CapturedTrial | None,
    bracket_before: tuple[float, float],
    bracket_after: tuple[float, float],
    decision: str,
) -> dict[str, Any]:
    phi0_ulp = math.ulp(phi0)
    phi_ulp = math.ulp(phi)
    phi_lo_ulp = math.ulp(phi_lo)
    armijo_ulp = math.ulp(armijo_rhs)
    phi0_denominator = max(phi0_ulp, phi_ulp)
    phi_lo_denominator = max(phi_lo_ulp, phi_ulp)
    armijo_denominator = max(armijo_ulp, phi_ulp)
    delta_phi0 = phi - phi0
    delta_phi_lo = phi - phi_lo
    delta_armijo = phi - armijo_rhs
    curvature_ratio = abs(dphi) / abs(dphi0)
    armijo_pass = phi <= armijo_rhs
    curvature_pass = abs(dphi) <= v4.CURVATURE_C2 * abs(dphi0)
    displacement = candidate - current
    changed = int(np.count_nonzero(candidate != current))
    candidate_equals_current = bool(np.array_equal(candidate, current))
    return {
        "optimizer_iteration": int(context["optimizer_iteration"]),
        "search_index": int(context["search_index"]),
        "trial_index": trial_index,
        "phase": phase,
        "phase_iteration": phase_iteration,
        "evaluations_before": evaluations_before,
        "evaluations_after": evaluations_after,
        "expansions_before": expansions_before,
        "expansions_after": expansions_after,
        "zoom_iterations_before": zoom_before,
        "zoom_iterations_after": zoom_after,
        "alpha": float(alpha),
        "alpha_hex": _float_hex(alpha),
        "phi0": float(phi0),
        "phi": float(phi),
        "phi_lo": float(phi_lo),
        "armijo_rhs": float(armijo_rhs),
        "phi0_hex": _float_hex(phi0),
        "phi_hex": _float_hex(phi),
        "phi_lo_hex": _float_hex(phi_lo),
        "armijo_rhs_hex": _float_hex(armijo_rhs),
        "phi_equal_phi0": bool(phi == phi0),
        "phi_equal_phi_lo": bool(phi == phi_lo),
        "phi_equal_armijo_rhs": bool(phi == armijo_rhs),
        "phi_minus_phi0": float(delta_phi0),
        "phi_minus_phi_lo": float(delta_phi_lo),
        "phi_minus_armijo_rhs": float(delta_armijo),
        "phi0_ulp": float(phi0_ulp),
        "phi_ulp": float(phi_ulp),
        "phi_lo_ulp": float(phi_lo_ulp),
        "armijo_rhs_ulp": float(armijo_ulp),
        "phi0_delta_ulp_denominator": float(phi0_denominator),
        "phi_lo_delta_ulp_denominator": float(phi_lo_denominator),
        "armijo_delta_ulp_denominator": float(armijo_denominator),
        "phi0_delta_in_ulps": _ratio_or_none(delta_phi0, phi0_denominator),
        "phi_lo_delta_in_ulps": _ratio_or_none(delta_phi_lo, phi_lo_denominator),
        "armijo_delta_in_ulps": _ratio_or_none(
            delta_armijo, armijo_denominator
        ),
        "dphi0": float(dphi0),
        "dphi": float(dphi),
        "dphi0_hex": _float_hex(dphi0),
        "dphi_hex": _float_hex(dphi),
        "dphi_abs_ratio": float(curvature_ratio),
        "dphi_sign": _sign(dphi),
        "dphi_nonnegative": bool(dphi >= 0.0),
        "armijo_pass": bool(armijo_pass),
        "curvature_pass": bool(curvature_pass),
        "combined_strong_wolfe_pass": bool(armijo_pass and curvature_pass),
        "objective_nondecrease_vs_phi_lo": bool(phi >= phi_lo),
        "derivative_bracket_sign_test": bool(
            dphi * (bracket_before[1] - bracket_before[0]) >= 0.0
        ),
        "bracket_before": _bracket(*bracket_before),
        "bracket_after": _bracket(*bracket_after),
        "decision": decision,
        "candidate_differs_from_current": not candidate_equals_current,
        "candidate_equals_current": candidate_equals_current,
        "changed_coordinate_count": changed,
        "candidate_displacement_l2": float(np.linalg.norm(displacement)),
        "candidate_displacement_linf": float(np.max(np.abs(displacement))),
        "candidate_equals_previous_trial": (
            None
            if previous_trial is None
            else bool(np.array_equal(candidate, previous_trial.candidate))
        ),
        "phi_equals_previous_trial": (
            None if previous_trial is None else bool(phi == previous_trial.value)
        ),
        "current_gradient_infinity_norm": float(context["gradient_inf"]),
        "gradient_infinity_tolerance": v4.GRADIENT_TOLERANCE,
        "gradient_tolerance_pass": bool(
            context["gradient_inf"] <= v4.GRADIENT_TOLERANCE
        ),
        "prior_relative_objective_change": _finite_or_none(
            float(context["prior_relative_change"])
        ),
        "prior_relative_change_was_infinite": not math.isfinite(
            float(context["prior_relative_change"])
        ),
        "relative_objective_tolerance": v4.RELATIVE_OBJECTIVE_TOLERANCE,
        "relative_objective_tolerance_pass": bool(
            context["prior_relative_change"]
            <= v4.RELATIVE_OBJECTIVE_TOLERANCE
        ),
        "both_convergence_criteria_pass_before_search": bool(
            context["gradient_inf"] <= v4.GRADIENT_TOLERANCE
            and context["prior_relative_change"]
            <= v4.RELATIVE_OBJECTIVE_TOLERANCE
        ),
        "current_objective": float(context["objective"]),
        "coefficient_l2_norm": float(context["coefficient_l2"]),
        "coefficient_linf_norm": float(context["coefficient_linf"]),
        "direction_l2_norm": float(context["direction_l2"]),
        "direction_linf_norm": float(context["direction_linf"]),
        "h0_scale": float(context["h0_scale"]),
        "history_pairs": int(context["history_pairs"]),
        "history_size_limit": v4.HISTORY_SIZE,
        "directional_derivative_initial": float(context["dphi0"]),
        "descent_direction": bool(context["dphi0"] < 0.0),
    }


def _trace_frozen_search(
    *,
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
    context: Mapping[str, Any],
) -> TracedSearch:
    """Run the untouched v4 search once, then reconstruct its scalar branches."""
    captured: list[CapturedTrial] = []

    def capture(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        candidate_value, candidate_gradient = evaluate(candidate)
        gradient_array = np.asarray(candidate_gradient, dtype=float)
        captured.append(
            CapturedTrial(
                candidate=np.asarray(candidate, dtype=float).copy(),
                value=float(candidate_value),
                gradient=gradient_array.copy(),
            )
        )
        return candidate_value, candidate_gradient

    actual_result = None
    actual_failure = None
    try:
        actual_result = v4._strong_wolfe_search_v4(
            vector, value, gradient, direction, capture
        )
    except v4.v3.StrongWolfeFailure as error:
        actual_failure = error

    derivative_initial = float(np.dot(gradient, direction))
    if not derivative_initial < 0.0:
        raise TraceDrift("trace was given a non-descent direction")
    events: list[dict[str, Any]] = []
    cursor = 0
    evaluations = 0
    expansions = 0
    zoom_iterations = 0
    previous_trial: CapturedTrial | None = None

    def consume(alpha: float) -> CapturedTrial:
        nonlocal cursor, evaluations
        if cursor >= len(captured):
            raise TraceDrift("captured v4 evaluations ended before reconstruction")
        trial = captured[cursor]
        expected = vector + alpha * direction
        if not np.array_equal(trial.candidate, expected):
            raise TraceDrift("captured candidate differs from frozen v4 arithmetic")
        cursor += 1
        evaluations += 1
        return trial

    def append_event(
        *,
        trial: CapturedTrial,
        phase: str,
        phase_iteration: int,
        alpha: float,
        phi_lo: float,
        bracket_before: tuple[float, float],
        bracket_after: tuple[float, float],
        decision: str,
        evaluations_before: int,
        expansions_before: int,
        zoom_before: int,
    ) -> None:
        nonlocal previous_trial
        dphi = float(np.dot(trial.gradient, direction))
        armijo_rhs = value + v4.ARMIJO_C1 * alpha * derivative_initial
        events.append(
            _trial_event(
                context=context,
                phase=phase,
                phase_iteration=phase_iteration,
                trial_index=len(events) + 1,
                evaluations_before=evaluations_before,
                evaluations_after=evaluations,
                expansions_before=expansions_before,
                expansions_after=expansions,
                zoom_before=zoom_before,
                zoom_after=zoom_iterations,
                alpha=alpha,
                phi0=value,
                phi=trial.value,
                phi_lo=phi_lo,
                armijo_rhs=armijo_rhs,
                dphi0=derivative_initial,
                dphi=dphi,
                candidate=trial.candidate,
                current=vector,
                previous_trial=previous_trial,
                bracket_before=bracket_before,
                bracket_after=bracket_after,
                decision=decision,
            )
        )
        previous_trial = trial

    def terminal_failure(message: str) -> dict[str, Any]:
        return {
            "kind": "failure",
            "message": message,
            "evaluations": evaluations,
            "expansions": expansions,
            "zoom_iterations": zoom_iterations,
        }

    def terminal_accept(alpha: float, trial: CapturedTrial) -> dict[str, Any]:
        dphi = float(np.dot(trial.gradient, direction))
        return {
            "kind": "accepted",
            "alpha": float(alpha),
            "value": trial.value,
            "dphi": dphi,
            "evaluations": evaluations,
            "expansions": expansions,
            "zoom_iterations": zoom_iterations,
            "candidate": trial.candidate,
            "gradient": trial.gradient,
        }

    def zoom(
        low: float,
        high: float,
        value_low: float,
        derivative_low: float,
    ) -> dict[str, Any]:
        nonlocal zoom_iterations
        for zoom_index in range(1, v4.ZOOM_MAXIMUM_ITERATIONS + 1):
            zoom_before = zoom_iterations
            zoom_iterations += 1
            alpha = v4.ZOOM_BISECTION_FRACTION * (low + high)
            if alpha == low or alpha == high:
                return terminal_failure(
                    "strong-Wolfe zoom bracket reached floating-point resolution"
                )
            evaluations_before = evaluations
            expansions_before = expansions
            trial = consume(alpha)
            phi = trial.value
            dphi = float(np.dot(trial.gradient, direction))
            armijo_rhs = value + v4.ARMIJO_C1 * alpha * derivative_initial
            before = (low, high)
            value_low_before = value_low
            if phi > armijo_rhs:
                high = alpha
                decision = "zoom_bracket_armijo_failure"
                after = (low, high)
            elif abs(dphi) <= v4.CURVATURE_C2 * abs(derivative_initial):
                decision = "accept_strong_wolfe"
                after = before
                append_event(
                    trial=trial,
                    phase="zoom",
                    phase_iteration=zoom_index,
                    alpha=alpha,
                    phi_lo=value_low,
                    bracket_before=before,
                    bracket_after=after,
                    decision=decision,
                    evaluations_before=evaluations_before,
                    expansions_before=expansions_before,
                    zoom_before=zoom_before,
                )
                return terminal_accept(alpha, trial)
            elif phi >= value_low:
                high = alpha
                decision = "zoom_bracket_objective_nondecrease"
                after = (low, high)
            else:
                sign_swap = dphi * (high - low) >= 0.0
                if sign_swap:
                    high = low
                low = alpha
                value_low = phi
                derivative_low = dphi
                decision = (
                    "zoom_shift_low_with_sign_swap"
                    if sign_swap
                    else "zoom_shift_low"
                )
                after = (low, high)
            append_event(
                trial=trial,
                phase="zoom",
                phase_iteration=zoom_index,
                alpha=alpha,
                phi_lo=value_low_before,
                bracket_before=before,
                bracket_after=after,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
        return terminal_failure("strong-Wolfe zoom iteration ceiling reached")

    previous_step = 0.0
    previous_value = value
    previous_derivative = derivative_initial
    alpha = v4.INITIAL_STEP
    outer_iteration = 0
    reconstructed: dict[str, Any]
    while True:
        evaluations_before = evaluations
        expansions_before = expansions
        zoom_before = zoom_iterations
        trial = consume(alpha)
        phi = trial.value
        dphi = float(np.dot(trial.gradient, direction))
        armijo_rhs = value + v4.ARMIJO_C1 * alpha * derivative_initial
        before = (previous_step, alpha)
        if phi > armijo_rhs:
            decision = "enter_zoom_armijo_failure"
            after = before
            append_event(
                trial=trial,
                phase="outer",
                phase_iteration=outer_iteration,
                alpha=alpha,
                phi_lo=previous_value,
                bracket_before=before,
                bracket_after=after,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
            reconstructed = zoom(
                previous_step, alpha, previous_value, previous_derivative
            )
            break
        if abs(dphi) <= v4.CURVATURE_C2 * abs(derivative_initial):
            decision = "accept_strong_wolfe"
            append_event(
                trial=trial,
                phase="outer",
                phase_iteration=outer_iteration,
                alpha=alpha,
                phi_lo=previous_value,
                bracket_before=before,
                bracket_after=before,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
            reconstructed = terminal_accept(alpha, trial)
            break
        if outer_iteration > 0 and phi >= previous_value:
            decision = "enter_zoom_objective_nondecrease"
            append_event(
                trial=trial,
                phase="outer",
                phase_iteration=outer_iteration,
                alpha=alpha,
                phi_lo=previous_value,
                bracket_before=before,
                bracket_after=before,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
            reconstructed = zoom(
                previous_step, alpha, previous_value, previous_derivative
            )
            break
        if dphi >= 0.0:
            decision = "enter_zoom_derivative_sign"
            after = (alpha, previous_step)
            append_event(
                trial=trial,
                phase="outer",
                phase_iteration=outer_iteration,
                alpha=alpha,
                phi_lo=previous_value,
                bracket_before=before,
                bracket_after=after,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
            reconstructed = zoom(alpha, previous_step, phi, dphi)
            break
        if alpha >= v4.MAXIMUM_STEP:
            decision = "fail_maximum_step"
            append_event(
                trial=trial,
                phase="outer",
                phase_iteration=outer_iteration,
                alpha=alpha,
                phi_lo=previous_value,
                bracket_before=before,
                bracket_after=before,
                decision=decision,
                evaluations_before=evaluations_before,
                expansions_before=expansions_before,
                zoom_before=zoom_before,
            )
            reconstructed = terminal_failure(
                "strong-Wolfe maximum step reached without a bracket"
            )
            break
        next_alpha = min(v4.EXPANSION_FACTOR * alpha, v4.MAXIMUM_STEP)
        previous_value_before = previous_value
        previous_step = alpha
        previous_value = phi
        previous_derivative = dphi
        expansions += 1
        outer_iteration += 1
        decision = "expand"
        append_event(
            trial=trial,
            phase="outer",
            phase_iteration=outer_iteration - 1,
            alpha=alpha,
            phi_lo=previous_value_before,
            bracket_before=before,
            bracket_after=(alpha, next_alpha),
            decision=decision,
            evaluations_before=evaluations_before,
            expansions_before=expansions_before,
            zoom_before=zoom_before,
        )
        alpha = next_alpha

    if cursor != len(captured):
        raise TraceDrift("reconstruction did not consume every frozen v4 evaluation")
    if reconstructed["kind"] == "accepted":
        if actual_result is None or actual_failure is not None:
            raise TraceDrift("reconstruction accepted but frozen v4 failed")
        if (
            reconstructed["alpha"] != actual_result.step
            or reconstructed["value"] != actual_result.value
            or reconstructed["evaluations"] != actual_result.evaluations
            or reconstructed["expansions"] != actual_result.expansions
            or reconstructed["zoom_iterations"] != actual_result.zoom_iterations
            or not np.array_equal(reconstructed["candidate"], actual_result.vector)
            or not np.array_equal(reconstructed["gradient"], actual_result.gradient)
        ):
            raise TraceDrift("reconstructed acceptance differs from frozen v4")
        summary = {
            "status": "accepted",
            "accepted_alpha": float(actual_result.step),
            "accepted_alpha_hex": _float_hex(actual_result.step),
            "evaluations": int(actual_result.evaluations),
            "expansions": int(actual_result.expansions),
            "zoom_iterations": int(actual_result.zoom_iterations),
            "armijo_slack": float(actual_result.armijo_slack),
            "curvature_ratio": float(actual_result.curvature_ratio),
            "accepted_both_exact_Wolfe_conditions": bool(
                actual_result.armijo_slack >= 0.0
                and actual_result.curvature_ratio <= v4.CURVATURE_C2
            ),
        }
    else:
        if actual_failure is None or actual_result is not None:
            raise TraceDrift("reconstruction failed but frozen v4 accepted")
        if (
            reconstructed["message"] != str(actual_failure)
            or reconstructed["evaluations"] != actual_failure.evaluations
            or reconstructed["expansions"] != actual_failure.expansions
            or reconstructed["zoom_iterations"] != actual_failure.zoom_iterations
        ):
            raise TraceDrift("reconstructed failure differs from frozen v4")
        summary = {
            "status": "failure",
            "message": str(actual_failure),
            "evaluations": int(actual_failure.evaluations),
            "expansions": int(actual_failure.expansions),
            "zoom_iterations": int(actual_failure.zoom_iterations),
        }
    summary.update(
        {
            "optimizer_iteration": int(context["optimizer_iteration"]),
            "search_index": int(context["search_index"]),
            "trial_count": len(events),
            "frozen_v4_reconstruction_exact": True,
        }
    )
    return TracedSearch(actual_result, actual_failure, events, summary)


def _optimizer_context(
    *,
    iteration: int,
    vector: np.ndarray,
    value: float,
    gradient: np.ndarray,
    direction: np.ndarray,
    h0_scale: float,
    history_pairs: int,
    relative_change: float,
) -> dict[str, Any]:
    return {
        "optimizer_iteration": iteration,
        "search_index": iteration,
        "gradient_inf": float(np.max(np.abs(gradient))),
        "prior_relative_change": relative_change,
        "objective": value,
        "coefficient_l2": float(np.linalg.norm(vector)),
        "coefficient_linf": float(np.max(np.abs(vector))),
        "direction_l2": float(np.linalg.norm(direction)),
        "direction_linf": float(np.max(np.abs(direction))),
        "h0_scale": h0_scale,
        "history_pairs": history_pairs,
        "dphi0": float(np.dot(gradient, direction)),
    }


def _synthetic_trace_self_test() -> dict[str, Any]:
    def zoom_objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        residual = float(point[0] - 0.5)
        return 1.0e16 + residual * residual, np.asarray([2.0 * residual])

    start = np.asarray([0.0])
    value, gradient = zoom_objective(start)
    direction = -gradient
    context = _optimizer_context(
        iteration=1,
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        h0_scale=1.0,
        history_pairs=0,
        relative_change=math.inf,
    )
    zoom_trace = _trace_frozen_search(
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        evaluate=zoom_objective,
        context=context,
    )
    if (
        zoom_trace.failure is not None
        or zoom_trace.result is None
        or zoom_trace.result.step != 0.5
        or zoom_trace.result.armijo_slack != 0.0
        or zoom_trace.result.curvature_ratio != 0.0
        or not any(event["phase"] == "zoom" for event in zoom_trace.events)
    ):
        raise AssertionError("synthetic zoom trace did not reproduce frozen v4")

    def outer_objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        scalar = float(point[0])
        return (
            float(2**55) + scalar * scalar / 36.0 - scalar,
            np.asarray([scalar / 18.0 - 1.0]),
        )

    start = np.asarray([0.0])
    value, gradient = outer_objective(start)
    direction = -gradient
    context = _optimizer_context(
        iteration=1,
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        h0_scale=1.0,
        history_pairs=0,
        relative_change=math.inf,
    )
    outer_trace = _trace_frozen_search(
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        evaluate=outer_objective,
        context=context,
    )
    if (
        outer_trace.failure is not None
        or outer_trace.result is None
        or outer_trace.result.step != 2.0
        or outer_trace.summary["expansions"] != 1
        or outer_trace.summary["zoom_iterations"] != 0
        or outer_trace.events[-1]["decision"] != "accept_strong_wolfe"
    ):
        raise AssertionError("synthetic outer trace did not reproduce frozen v4")

    def ceiling_evaluator(point: np.ndarray) -> tuple[float, np.ndarray]:
        # Deliberately adversarial evaluator used only to exercise the exact
        # fail-closed trace path; it is not a scientific objective.
        return 0.0, np.asarray([-1.0])

    start = np.asarray([0.0])
    value = 0.0
    gradient = np.asarray([-1.0])
    direction = np.asarray([1.0])
    context = _optimizer_context(
        iteration=1,
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        h0_scale=1.0,
        history_pairs=0,
        relative_change=math.inf,
    )
    ceiling_trace = _trace_frozen_search(
        vector=start,
        value=value,
        gradient=gradient,
        direction=direction,
        evaluate=ceiling_evaluator,
        context=context,
    )
    if (
        ceiling_trace.result is not None
        or ceiling_trace.failure is None
        or ceiling_trace.failure.evaluations != EXPECTED_FAILURE_EVALUATIONS
        or ceiling_trace.failure.zoom_iterations
        != EXPECTED_FAILURE_ZOOM_ITERATIONS
        or ceiling_trace.summary["trial_count"] != EXPECTED_FAILURE_EVALUATIONS
    ):
        raise AssertionError("synthetic zoom-ceiling failure trace changed")
    return {
        "status": "pass",
        "uses_target_data": False,
        "zoom": {
            "accepted_alpha": zoom_trace.summary["accepted_alpha"],
            "events": zoom_trace.summary["trial_count"],
            "reconstruction_exact": True,
        },
        "outer_expansion": {
            "accepted_alpha": outer_trace.summary["accepted_alpha"],
            "events": outer_trace.summary["trial_count"],
            "expansions": outer_trace.summary["expansions"],
            "reconstruction_exact": True,
        },
        "zoom_ceiling_failure": {
            "evaluations": ceiling_trace.summary["evaluations"],
            "zoom_iterations": ceiling_trace.summary["zoom_iterations"],
            "events": ceiling_trace.summary["trial_count"],
            "reconstruction_exact": True,
        },
        "vectors_emitted": False,
    }


def _validate_output_root(path: Path, scratch: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise base.ModelError("real trace must run in a Slurm job")
    episode_root = v4.v3._require_unsymlinked_directory(
        scratch / base.EPISODE_SCRATCH_RELATIVE, "EP10 Scratch root"
    )
    absolute = Path(os.path.abspath(path))
    expected = episode_root / f"{OUTPUT_BASENAME_PREFIX}{job_id}"
    if (
        not path.is_absolute()
        or absolute != expected
        or not absolute.is_dir()
        or absolute.is_symlink()
        or absolute != absolute.resolve(strict=True)
        or absolute.stat().st_uid != os.getuid()
    ):
        raise base.ModelError("trace output is not the exact owned job-scoped directory")
    names = sorted(item.name for item in absolute.iterdir())
    if names != [RUNTIME_LOG_BASENAME]:
        raise base.ModelError("trace output must initially contain only runtime.log")
    runtime_log = absolute / RUNTIME_LOG_BASENAME
    if (
        not runtime_log.is_file()
        or runtime_log.is_symlink()
        or runtime_log != runtime_log.resolve(strict=True)
        or runtime_log.stat().st_uid != os.getuid()
    ):
        raise base.ModelError("trace runtime.log failed its ownership/path seal")
    return absolute


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    _require_exact_unsymlinked_file(
        SCRIPT_PATH,
        SCRIPT_DIR / "trace_v3_m0_v4_lambda1000.py",
        "trace-v1 script",
    )
    v4.validate_solver_addendum(args.contract, args.solver_addendum)
    scratch, target_root, interface_root = v4.v3._validate_real_input_roots(
        args.target_root, args.interface_root
    )
    output_root = _validate_output_root(args.output_root, scratch)
    output_path = output_root / OUTPUT_JSON_BASENAME
    payload: dict[str, Any] = {
        "status": "fail",
        "stage": "v3_M0_solver_v4_lambda1000_trace_v1",
        "solver_addendum": args.solver_addendum.name,
        "ridge_lambda": RIDGE_LAMBDA,
        "zero_start": True,
        "single_lambda_only": True,
        "ridge_path_run": False,
        "prediction_or_scoring_run": False,
        "held_out_scores_emitted": False,
        "coefficient_gradient_prediction_or_contrast_vectors_emitted": False,
        "row_ids_emitted": False,
        "final_or_unassigned_opened": False,
        "output_root": str(output_root),
    }
    try:
        synthetic = _synthetic_trace_self_test()
        records = base.load_real_records(target_root, interface_root)
        training = [
            record
            for record in records
            if record.group not in (OUTER_HELD_GROUP, INNER_HELD_GROUP)
        ]
        training_groups = set(record.group for record in training)
        if (
            len(training) != EXPECTED_TRAINING_CELLS
            or len(training_groups) != EXPECTED_TRAINING_GROUPS
            or OUTER_HELD_GROUP in training_groups
            or INNER_HELD_GROUP in training_groups
        ):
            raise TraceDrift("exact trace training split changed")

        transform = base.fit_design_transform(training)
        x = base.build_design(training, transform)
        y, k_values, groups = base.outcome_matrix(training)
        target_contrasts = base.helmert(16)
        dimension = x.shape[1] * target_contrasts.shape[1]
        if dimension != v4.PARAMETER_DIMENSION:
            raise TraceDrift("trace parameter dimension changed")

        def evaluate(candidate: np.ndarray) -> tuple[float, np.ndarray]:
            return base.objective_and_gradient(
                candidate,
                x,
                y,
                k_values,
                groups,
                target_contrasts,
                RIDGE_LAMBDA,
            )

        vector = np.zeros(dimension, dtype=float)
        value, gradient = evaluate(vector)
        initial_gradient_inf = float(np.max(np.abs(gradient)))
        if initial_gradient_inf <= v4.GRADIENT_TOLERANCE:
            raise TraceDrift(
                "frozen v4 would converge at the initial value before tracing"
            )
        s_history: list[np.ndarray] = []
        y_history: list[np.ndarray] = []
        relative_change = math.inf
        all_events: list[dict[str, Any]] = []
        searches: list[dict[str, Any]] = []
        optimizer_iterations: list[dict[str, Any]] = []
        payload["trial_events"] = all_events
        payload["searches"] = searches
        payload["optimizer_iterations"] = optimizer_iterations
        terminal_failure = None

        for iteration in range(1, 4):
            direction, h0_scale = v4.v2._lbfgs_direction_v2(
                gradient, s_history, y_history
            )
            context = _optimizer_context(
                iteration=iteration,
                vector=vector,
                value=value,
                gradient=gradient,
                direction=direction,
                h0_scale=h0_scale,
                history_pairs=len(s_history),
                relative_change=relative_change,
            )
            traced = _trace_frozen_search(
                vector=vector,
                value=value,
                gradient=gradient,
                direction=direction,
                evaluate=evaluate,
                context=context,
            )
            all_events.extend(traced.events)
            searches.append(traced.summary)
            iteration_record: dict[str, Any] = {
                "optimizer_iteration": iteration,
                "objective_before": float(value),
                "gradient_infinity_norm_before": float(
                    np.max(np.abs(gradient))
                ),
                "prior_relative_objective_change": _finite_or_none(
                    relative_change
                ),
                "prior_relative_change_was_infinite": not math.isfinite(
                    relative_change
                ),
                "coefficient_l2_norm_before": float(np.linalg.norm(vector)),
                "direction_l2_norm": float(np.linalg.norm(direction)),
                "h0_scale": float(h0_scale),
                "history_pairs_before": len(s_history),
                "directional_derivative_initial": float(
                    np.dot(gradient, direction)
                ),
                "descent_direction": bool(np.dot(gradient, direction) < 0.0),
                "search_status": traced.summary["status"],
            }
            if traced.failure is not None:
                iteration_record.update(
                    {
                        "accepted": False,
                        "failure_evaluations": traced.failure.evaluations,
                        "failure_expansions": traced.failure.expansions,
                        "failure_zoom_iterations": traced.failure.zoom_iterations,
                    }
                )
                optimizer_iterations.append(iteration_record)
                terminal_failure = traced.failure
                if iteration != 3:
                    raise TraceDrift(
                        f"line search failed at optimizer iteration {iteration}, expected 3"
                    )
                break

            if traced.result is None:
                raise TraceDrift("trace search returned neither acceptance nor failure")
            expected_step = EXPECTED_ACCEPTED_STEPS[iteration - 1]
            if iteration > 2 or traced.result.step != expected_step:
                raise TraceDrift(
                    f"accepted step drift at iteration {iteration}: "
                    f"observed {traced.result.step.hex()}"
                )
            new_vector = traced.result.vector
            new_value = traced.result.value
            new_gradient = traced.result.gradient
            s_value = new_vector - vector
            y_value = new_gradient - gradient
            pair_curvature = float(np.dot(s_value, y_value))
            curvature_floor = (
                v4.CURVATURE_FLOOR_MULTIPLIER
                * max(1.0, float(np.linalg.norm(s_value)))
                * max(1.0, float(np.linalg.norm(y_value)))
            )
            pair_accepted = pair_curvature > curvature_floor
            if pair_accepted:
                s_history.append(s_value)
                y_history.append(y_value)
                if len(s_history) > v4.HISTORY_SIZE:
                    del s_history[0]
                    del y_history[0]
            relative_change = abs(new_value - value) / max(
                1.0, abs(value), abs(new_value)
            )
            vector, value, gradient = new_vector, new_value, new_gradient
            gradient_inf = float(np.max(np.abs(gradient)))
            iteration_record.update(
                {
                    "accepted": True,
                    "accepted_alpha": float(traced.result.step),
                    "accepted_alpha_hex": _float_hex(traced.result.step),
                    "objective_after": float(value),
                    "gradient_infinity_norm_after": gradient_inf,
                    "relative_objective_change_after": float(relative_change),
                    "gradient_tolerance_pass_after": bool(
                        gradient_inf <= v4.GRADIENT_TOLERANCE
                    ),
                    "relative_tolerance_pass_after": bool(
                        relative_change <= v4.RELATIVE_OBJECTIVE_TOLERANCE
                    ),
                    "both_convergence_criteria_pass_after": bool(
                        gradient_inf <= v4.GRADIENT_TOLERANCE
                        and relative_change <= v4.RELATIVE_OBJECTIVE_TOLERANCE
                    ),
                    "coefficient_l2_norm_after": float(np.linalg.norm(vector)),
                    "pair_curvature": pair_curvature,
                    "curvature_floor": curvature_floor,
                    "curvature_pair_accepted": pair_accepted,
                    "history_pairs_after": len(s_history),
                }
            )
            optimizer_iterations.append(iteration_record)
            if (
                gradient_inf <= v4.GRADIENT_TOLERANCE
                and relative_change <= v4.RELATIVE_OBJECTIVE_TOLERANCE
            ):
                raise TraceDrift(
                    "frozen v4 would converge before the expected third search"
                )

        if terminal_failure is None:
            raise TraceDrift("third v4 search unexpectedly accepted")
        if (
            len(searches) != 3
            or [item.get("accepted_alpha") for item in searches[:2]]
            != list(EXPECTED_ACCEPTED_STEPS)
            or searches[2]["status"] != "failure"
            or terminal_failure.evaluations != EXPECTED_FAILURE_EVALUATIONS
            or terminal_failure.zoom_iterations
            != EXPECTED_FAILURE_ZOOM_ITERATIONS
        ):
            raise TraceDrift("lambda-1000 terminal trace signature changed")

        payload.update(
            {
                "status": "pass",
                "training_cells": len(training),
                "training_groups": len(training_groups),
                "outer_group_excluded": True,
                "inner_group_excluded": True,
                "parameter_dimension": dimension,
                "synthetic_trace_self_test": synthetic,
                "expected_signature": {
                    "accepted_steps": list(EXPECTED_ACCEPTED_STEPS),
                    "third_search_failure_evaluations": EXPECTED_FAILURE_EVALUATIONS,
                    "third_search_failure_zoom_iterations": EXPECTED_FAILURE_ZOOM_ITERATIONS,
                },
                "optimizer_iterations": optimizer_iterations,
                "searches": searches,
                "trial_events": all_events,
                "observed_signature_pass": True,
            }
        )
        base._atomic_json(output_path, payload)
        return payload
    except Exception as error:
        error_code = (
            "trace_trajectory_drift"
            if isinstance(error, TraceDrift)
            else "trace_failed_closed"
        )
        payload.update(
            {
                "status": "fail",
                "error_code": error_code,
                "raw_error_emitted": False,
            }
        )
        base._atomic_json(output_path, payload)
        raise TraceDrift(error_code) from None


def run_self_test(contract: Path, solver_addendum: Path) -> dict[str, Any]:
    _require_exact_unsymlinked_file(
        SCRIPT_PATH,
        SCRIPT_DIR / "trace_v3_m0_v4_lambda1000.py",
        "trace-v1 script",
    )
    v4.validate_solver_addendum(contract, solver_addendum)
    result = _synthetic_trace_self_test()
    return {
        "status": "pass",
        "stage": "v3_M0_solver_v4_trace_v1_synthetic_self_test",
        "synthetic_trace": result,
        "uses_real_data": False,
        "vectors_emitted": False,
        "final_or_unassigned_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--trace", action="store_true")
    parser.add_argument("--contract", type=Path, default=BASE_CONTRACT_PATH)
    parser.add_argument("--solver-addendum", type=Path, default=V4_ADDENDUM_PATH)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--interface-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if any(
            value is not None
            for value in (args.target_root, args.interface_root, args.output_root)
        ):
            parser.error("--self-test does not accept real input/output paths")
        try:
            result = run_self_test(args.contract, args.solver_addendum)
        except (base.ModelError, AssertionError, ArithmeticError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0

    missing = [
        name
        for name in ("target_root", "interface_root", "output_root")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error(
            "--trace requires: "
            + ", ".join("--" + item.replace("_", "-") for item in missing)
        )
    try:
        result = run_trace(args)
    except Exception:
        print("ERROR: trace_failed_closed", file=sys.stderr)
        return 2
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
