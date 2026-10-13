"""Frozen uncertainty calculation for EP12 mode-support diagnostics.

This module is deliberately data-source agnostic.  Its rows are *whole provider
types*, never cells or individual directions.  A single bootstrap draw resamples
those rows once and applies the same indices to both prediction directions and
to every diagnostic.  Consequently callers cannot accidentally turn the paired
bilateral calculation into two easier, unpaired tests.

No final-connectivity access or filesystem discovery occurs here.  The caller
must supply an already authenticated development-only result table hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


BOOTSTRAP_REPLICATES = 1_999
CONFIDENCE_LEVEL = 0.95
PROTOCOL = "ep12.whole_provider_type_paired_simultaneous_max_t.v1"
SCHEMA_VERSION = "ep12.mode_support_inference.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModeSupportInferenceError(ValueError):
    """Raised when supplied evidence violates the frozen inference contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModeSupportInferenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _array(
    name: str,
    value: Any,
    *,
    rows: int | None = None,
    directions: bool = False,
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModeSupportInferenceError(f"{name} must be a numeric array") from exc
    if result.ndim == 0:
        raise ModeSupportInferenceError(f"{name} must not be scalar")
    if rows is not None and (result.ndim < 1 or result.shape[0] != rows):
        raise ModeSupportInferenceError(
            f"{name} must have one row for each whole provider type"
        )
    if directions and (result.ndim < 2 or result.shape[1] != 2):
        raise ModeSupportInferenceError(
            f"{name} must contain exactly the two paired prediction directions"
        )
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ModeSupportInferenceError(f"{name} must be nonempty and finite")
    return result


def _as_metric_cube(name: str, value: Any, rows: int) -> np.ndarray:
    result = _array(name, value, rows=rows, directions=True)
    if result.ndim == 2:
        result = result[:, :, np.newaxis]
    if result.ndim != 3:
        raise ModeSupportInferenceError(f"{name} must have shape (type, 2, term)")
    return result


def _as_cross_side_matrix(name: str, value: Any, rows: int) -> np.ndarray:
    result = _array(name, value, rows=rows)
    if result.ndim == 1:
        result = result[:, np.newaxis]
    if result.ndim != 2:
        raise ModeSupportInferenceError(f"{name} must have shape (type, match)")
    return result


def _seed(contract_sha256: str, result_table_sha256: str) -> int:
    seed_material = _canonical_bytes(
        {
            "protocol": PROTOCOL,
            "contract_sha256": contract_sha256,
            "result_table_sha256": result_table_sha256,
        }
    )
    # PCG64 consumes an unsigned integer.  Sixteen digest bytes retain ample
    # entropy while remaining stable across Python and NumPy versions.
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "big")


def _finalize(core: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(core)
    result["inference_sha256"] = _digest(core)
    return result


def verify_mode_support_result(result: Mapping[str, Any]) -> None:
    """Authenticate a previously produced inference result, failing closed."""

    if not isinstance(result, Mapping):
        raise ModeSupportInferenceError("mode-support result must be a mapping")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ModeSupportInferenceError("unsupported mode-support result schema")
    claimed = result.get("inference_sha256")
    _require_sha256("inference_sha256", claimed)
    core = {key: value for key, value in result.items() if key != "inference_sha256"}
    if _digest(core) != claimed:
        raise ModeSupportInferenceError("mode-support result digest mismatch")
    if result.get("protocol") != PROTOCOL:
        raise ModeSupportInferenceError("mode-support protocol mismatch")
    bootstrap = result.get("bootstrap")
    if result.get("complete_input"):
        if not isinstance(bootstrap, Mapping):
            raise ModeSupportInferenceError("complete result lacks bootstrap metadata")
        if bootstrap.get("replicates") != BOOTSTRAP_REPLICATES:
            raise ModeSupportInferenceError("bootstrap replicate count is not frozen")
        if bootstrap.get("directions_resampled_together") is not True:
            raise ModeSupportInferenceError("directions were not resampled together")
        if bootstrap.get("unit") != "whole_provider_type":
            raise ModeSupportInferenceError("bootstrap unit is not whole provider type")
        inputs = result.get("whole_provider_type_inputs")
        if not isinstance(inputs, Mapping):
            raise ModeSupportInferenceError("complete result lacks replayable whole-type inputs")
        if _digest(inputs) != result.get("input_binding_sha256"):
            raise ModeSupportInferenceError("whole-type input binding digest mismatch")
        expected_seed = _seed(
            _require_sha256("contract_sha256", result.get("contract_sha256")),
            _require_sha256(
                "result_table_sha256", result.get("result_table_sha256")
            ),
        )
        if bootstrap.get("seed") != expected_seed:
            raise ModeSupportInferenceError("bootstrap seed binding mismatch")
    if result.get("final_connectivity_accessed") is not False:
        raise ModeSupportInferenceError("mode-support artifact is not development-only")


def infer_mode_support(
    *,
    comparison_trial_id: str,
    configuration_sha256: str,
    contract_sha256: str,
    result_table_sha256: str,
    meaningful_margin: float,
    prevalence_parameter_rank_definition: str,
    whole_provider_type_ids: Sequence[str],
    prediction_gain_by_type_direction: Any | None,
    component_symmetric_kl_by_type_direction: Any | None,
    matched_contrast_cosine_by_type: Any | None,
    partner_contrast_by_type_direction: Any | None,
    effective_memberships_by_direction_component: Any | None,
    fitted_parameter_ranks_by_direction_component: Any | None,
) -> dict[str, Any]:
    """Calculate the frozen, simultaneous mode-support uncertainty report.

    ``component_symmetric_kl_by_type_direction`` may contain one or more
    component-pair terms.  Every term must have a simultaneous lower bound
    above zero.  ``matched_contrast_cosine_by_type`` similarly contains one or
    more label-invariant Hungarian matches.  Partner contrasts have shape
    ``(provider_type, direction, partner_term)``; at least one term must exclude
    zero with the same sign in both source fits.

    Missing scientific inputs produce an authenticated unresolved artifact.
    Malformed or nonfinite supplied inputs are technical failures and raise.
    """

    if not isinstance(comparison_trial_id, str) or not comparison_trial_id:
        raise ModeSupportInferenceError("comparison_trial_id must be nonempty")
    _require_sha256("configuration_sha256", configuration_sha256)
    _require_sha256("contract_sha256", contract_sha256)
    _require_sha256("result_table_sha256", result_table_sha256)
    try:
        margin = float(meaningful_margin)
    except (TypeError, ValueError) as exc:
        raise ModeSupportInferenceError("meaningful_margin must be finite") from exc
    if not math.isfinite(margin):
        raise ModeSupportInferenceError("meaningful_margin must be finite")
    if (
        not isinstance(prevalence_parameter_rank_definition, str)
        or not prevalence_parameter_rank_definition
    ):
        raise ModeSupportInferenceError(
            "prevalence_parameter_rank_definition must be frozen and nonempty"
        )

    if isinstance(whole_provider_type_ids, (str, bytes)):
        raise ModeSupportInferenceError("whole_provider_type_ids must be a sequence")
    type_ids = [str(value) for value in whole_provider_type_ids]
    if any(not value for value in type_ids) or len(set(type_ids)) != len(type_ids):
        raise ModeSupportInferenceError(
            "whole provider type identifiers must be nonempty and unique"
        )

    supplied = {
        "prediction_gain_by_type_direction": prediction_gain_by_type_direction,
        "component_symmetric_kl_by_type_direction": (
            component_symmetric_kl_by_type_direction
        ),
        "matched_contrast_cosine_by_type": matched_contrast_cosine_by_type,
        "partner_contrast_by_type_direction": partner_contrast_by_type_direction,
        "effective_memberships_by_direction_component": (
            effective_memberships_by_direction_component
        ),
        "fitted_parameter_ranks_by_direction_component": (
            fitted_parameter_ranks_by_direction_component
        ),
    }
    missing = sorted(name for name, value in supplied.items() if value is None)
    if len(type_ids) < 2:
        missing.append("at_least_two_whole_provider_types")
    if missing:
        core = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "comparison_trial_id": comparison_trial_id,
            "configuration_sha256": configuration_sha256,
            "contract_sha256": contract_sha256,
            "result_table_sha256": result_table_sha256,
            "meaningful_margin": margin,
            "prevalence_parameter_rank_definition": (
                prevalence_parameter_rank_definition
            ),
            "complete_input": False,
            "missing_inputs": sorted(set(missing)),
            "whole_provider_type_count": len(type_ids),
            "bootstrap": None,
            "diagnostics": None,
            "gates": {
                "prediction_gain": False,
                "component_separation": False,
                "prevalence": False,
                "cross_side_matching": False,
                "partner_identifiability": False,
            },
            "scientific_mode_support": False,
            "support_state": "unresolved",
            "null_eligible_from_mode_support": False,
            "final_connectivity_accessed": False,
        }
        return _finalize(core)

    n_types = len(type_ids)
    gain = _array(
        "prediction_gain_by_type_direction",
        prediction_gain_by_type_direction,
        rows=n_types,
        directions=True,
    )
    if gain.ndim != 2:
        raise ModeSupportInferenceError(
            "prediction_gain_by_type_direction must have shape (type, 2)"
        )
    kl = _as_metric_cube(
        "component_symmetric_kl_by_type_direction",
        component_symmetric_kl_by_type_direction,
        n_types,
    )
    if np.any(kl < 0.0):
        raise ModeSupportInferenceError("symmetric KL contributions cannot be negative")
    cosine = _as_cross_side_matrix(
        "matched_contrast_cosine_by_type",
        matched_contrast_cosine_by_type,
        n_types,
    )
    if np.any(cosine < -1.0) or np.any(cosine > 1.0):
        raise ModeSupportInferenceError("matched contrast cosine must be in [-1, 1]")
    partner = _as_metric_cube(
        "partner_contrast_by_type_direction",
        partner_contrast_by_type_direction,
        n_types,
    )

    memberships = _array(
        "effective_memberships_by_direction_component",
        effective_memberships_by_direction_component,
        rows=n_types,
        directions=True,
    )
    ranks = _array(
        "fitted_parameter_ranks_by_direction_component",
        fitted_parameter_ranks_by_direction_component,
        rows=n_types,
        directions=True,
    )
    if memberships.ndim != 3 or ranks.shape != memberships.shape:
        raise ModeSupportInferenceError(
            "effective memberships and fitted ranks must have identical "
            "(whole_type, 2, component) shape"
        )
    if np.any(memberships < 0.0):
        raise ModeSupportInferenceError("effective memberships cannot be negative")
    if np.any(ranks <= 0.0) or np.any(ranks != np.floor(ranks)):
        raise ModeSupportInferenceError("fitted parameter ranks must be positive integers")

    flattened = [
        gain,
        kl.reshape(n_types, -1),
        cosine,
        partner.reshape(n_types, -1),
    ]
    widths = [value.shape[1] for value in flattened]
    matrix = np.concatenate(flattened, axis=1)
    estimates = matrix.mean(axis=0)
    standard_errors = matrix.std(axis=0, ddof=1) / math.sqrt(n_types)

    seed = _seed(contract_sha256, result_table_sha256)
    rng = np.random.Generator(np.random.PCG64(seed))
    # The same index matrix is used for every coordinate, preserving both the
    # source-direction pairing and all between-diagnostic dependence.
    indices = rng.integers(0, n_types, size=(BOOTSTRAP_REPLICATES, n_types))
    bootstrap_means = matrix[indices].mean(axis=1)
    variable = standard_errors > 0.0
    if np.any(variable):
        studentized = np.abs(
            (bootstrap_means[:, variable] - estimates[variable])
            / standard_errors[variable]
        )
        max_t = np.max(studentized, axis=1)
    else:
        max_t = np.zeros(BOOTSTRAP_REPLICATES, dtype=np.float64)
    critical = float(np.quantile(max_t, CONFIDENCE_LEVEL, method="higher"))
    lower = estimates - critical * standard_errors
    upper = estimates + critical * standard_errors

    cursor = 0
    slices: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for width in widths:
        slices.append(
            (
                estimates[cursor : cursor + width],
                lower[cursor : cursor + width],
                upper[cursor : cursor + width],
            )
        )
        cursor += width
    gain_e, gain_l, gain_u = slices[0]
    kl_e, kl_l, kl_u = slices[1]
    cosine_e, cosine_l, cosine_u = slices[2]
    partner_e, partner_l, partner_u = slices[3]
    gain_e = gain_e.reshape(gain.shape[1:])
    gain_l = gain_l.reshape(gain.shape[1:])
    gain_u = gain_u.reshape(gain.shape[1:])
    kl_e = kl_e.reshape(kl.shape[1:])
    kl_l = kl_l.reshape(kl.shape[1:])
    kl_u = kl_u.reshape(kl.shape[1:])
    cosine_e = cosine_e.reshape(cosine.shape[1:])
    cosine_l = cosine_l.reshape(cosine.shape[1:])
    cosine_u = cosine_u.reshape(cosine.shape[1:])
    partner_e = partner_e.reshape(partner.shape[1:])
    partner_l = partner_l.reshape(partner.shape[1:])
    partner_u = partner_u.reshape(partner.shape[1:])

    gain_gate = bool(np.all(gain_l > margin))
    separation_gate = bool(np.all(kl_l > 0.0))
    cosine_gate = bool(np.all(cosine_l > 0.0))
    prevalence_gate = bool(np.all(memberships > ranks))
    same_sign_terms: list[int] = []
    for term in range(partner.shape[2]):
        both_positive = bool(np.all(partner_l[:, term] > 0.0))
        both_negative = bool(np.all(partner_u[:, term] < 0.0))
        if both_positive or both_negative:
            same_sign_terms.append(term)
    partner_gate = bool(same_sign_terms)

    gates = {
        "prediction_gain": gain_gate,
        "component_separation": separation_gate,
        "prevalence": prevalence_gate,
        "cross_side_matching": cosine_gate,
        "partner_identifiability": partner_gate,
    }
    supported = all(gates.values())
    input_binding = {
        "whole_provider_type_ids": type_ids,
        "prediction_gain_by_type_direction": gain.tolist(),
        "component_symmetric_kl_by_type_direction": kl.tolist(),
        "matched_contrast_cosine_by_type": cosine.tolist(),
        "partner_contrast_by_type_direction": partner.tolist(),
        "effective_memberships_by_direction_component": memberships.tolist(),
        "fitted_parameter_ranks_by_direction_component": ranks.astype(int).tolist(),
    }
    core = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "comparison_trial_id": comparison_trial_id,
        "configuration_sha256": configuration_sha256,
        "contract_sha256": contract_sha256,
        "result_table_sha256": result_table_sha256,
        "meaningful_margin": margin,
        "prevalence_parameter_rank_definition": prevalence_parameter_rank_definition,
        "complete_input": True,
        "missing_inputs": [],
        "whole_provider_type_count": n_types,
        "input_binding_sha256": _digest(input_binding),
        "whole_provider_type_inputs": input_binding,
        "bootstrap": {
            "unit": "whole_provider_type",
            "directions_resampled_together": True,
            "interval": "paired_simultaneous_95_percent_max_t",
            "confidence_level": CONFIDENCE_LEVEL,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_derivation": "sha256_of_contract_and_result_table",
            "seed": seed,
            "simultaneous_coordinate_count": int(matrix.shape[1]),
            "critical_value": critical,
        },
        "diagnostics": {
            "prediction_gain": {
                "estimate_by_direction": gain_e.tolist(),
                "simultaneous_lower_by_direction": gain_l.tolist(),
                "simultaneous_upper_by_direction": gain_u.tolist(),
                "strict_threshold": margin,
                "passed": gain_gate,
            },
            "component_symmetric_kl": {
                "estimate_by_direction_term": kl_e.tolist(),
                "simultaneous_lower_by_direction_term": kl_l.tolist(),
                "simultaneous_upper_by_direction_term": kl_u.tolist(),
                "strict_threshold": 0.0,
                "passed": separation_gate,
            },
            "matched_contrast_cosine": {
                "estimate_by_match": cosine_e.tolist(),
                "simultaneous_lower_by_match": cosine_l.tolist(),
                "simultaneous_upper_by_match": cosine_u.tolist(),
                "strict_threshold": 0.0,
                "passed": cosine_gate,
            },
            "partner_contrast": {
                "estimate_by_direction_term": partner_e.tolist(),
                "simultaneous_lower_by_direction_term": partner_l.tolist(),
                "simultaneous_upper_by_direction_term": partner_u.tolist(),
                "same_sign_terms_excluding_zero_in_both_source_fits": same_sign_terms,
                "passed": partner_gate,
            },
            "prevalence": {
                "effective_memberships_by_direction_component": memberships.tolist(),
                "fitted_parameter_ranks_by_direction_component": (
                    ranks.astype(int).tolist()
                ),
                "rule": "effective_membership_strictly_exceeds_fitted_parameter_rank",
                "passed": prevalence_gate,
            },
        },
        "gates": gates,
        "scientific_mode_support": supported,
        # A failed interval is not evidence for a no-mode ontology.  EP12 calls
        # it unresolved and simply declines to launch the expensive null.
        "support_state": "supported" if supported else "unresolved",
        "null_eligible_from_mode_support": supported,
        "final_connectivity_accessed": False,
    }
    return _finalize(core)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CONFIDENCE_LEVEL",
    "ModeSupportInferenceError",
    "PROTOCOL",
    "SCHEMA_VERSION",
    "infer_mode_support",
    "verify_mode_support_result",
]
