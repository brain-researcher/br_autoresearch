"""Frozen branch definitions and pure mechanics for EP12 production controls.

The helpers in this module never decide whether a biological effect is robust
or supported.  A composite control passes only when its entire prespecified
branch inventory executed, every applicable conservation/capacity check
passed, and every branch is bound to a complete T/U/M result.  No magnitude
threshold is introduced here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .policy import digest_object


CONTROL_SPEC_SCHEMA = "ep12.falsifier_control_implementation_spec.v1"
CONTROL_RECEIPT_SCHEMA = "ep12.composite_falsifier_control_receipt.v1"
BRANCH_MANIFEST_SCHEMA = "ep12.falsifier_control_branch_manifest.v1"
MODEL_TRIPLET = ("T_type_template", "U_flexible_unimodal", "M_residual_modes")

COUNTABLE_COMPOSITE_CONTROLS = (
    "cross_side_component_alignment_permutation",
    "binary_weighted_vocabulary_and_rank_sensitivities",
    "nuisance_and_partner_block_ablations",
    "capacity_matched_noise_controls",
    "leave_one_development_type_and_partner_family_influence",
    "few_type_and_high_strength_neuron_concentration_check",
    "status_endpoint_anatomy_and_missingness_checks",
)

ENDPOINT_BINS = (
    "other_typed",
    "unknown",
    "untyped",
    "fragment",
    "proofreading",
    "missing_annotation",
    "out_of_vocabulary",
)


class FalsifierControlError(ValueError):
    """A frozen control branch is unavailable, incomplete, or inconsistent."""


def implementation_spec() -> dict[str, Any]:
    """Return the complete threshold-free production definition."""

    controls = {
        "cross_side_component_alignment_permutation": {
            "branch_inventory": ["unpermuted", "deterministic_within_K_permutation"],
            "input": "persisted source-side M component partner contrasts",
            "strata": "fitted component count K",
            "score": "maximum-mean cosine over every component permutation",
            "target_side_refit": False,
        },
        "binary_weighted_vocabulary_and_rank_sensitivities": {
            "required_branches": ["weighted_baseline", "binary"],
            "expanded_branches": (
                "Cartesian product of every SEARCH_POLICY partner_vocabulary and rank"
            ),
            "common_models": list(MODEL_TRIPLET),
        },
        "nuisance_and_partner_block_ablations": {
            "nuisance_branches": "every nonempty declared nuisance block",
            "partner_branches": (
                "every nonempty subclass and class family in partner_type_hierarchy"
            ),
            "partner_operation": "pool family into an explicit retained-mass bin",
            "common_models": list(MODEL_TRIPLET),
        },
        "capacity_matched_noise_controls": {
            "required_branches": ["weighted_baseline", "pooled_multinomial_noise"],
            "noise": (
                "deterministic pooled multinomial with original matrix shape and "
                "each original row total"
            ),
            "capacity": "byte-identical scientific configuration across branches",
        },
        "leave_one_development_type_and_partner_family_influence": {
            "type_branches": (
                "analytic leave-one for every complete equal-weight provider-type record"
            ),
            "partner_branches": (
                "every nonempty subclass and class family in partner_type_hierarchy"
            ),
            "common_models": list(MODEL_TRIPLET),
        },
        "few_type_and_high_strength_neuron_concentration_check": {
            "type_curve": "complete leave-one-provider-type curve",
            "neuron_curve": "complete leave-one-target-neuron curve",
            "strength_curve": (
                "descending-strength cumulative deletion until one neuron remains in "
                "each provider-type/direction cell"
            ),
            "acceptance_cutoff": None,
        },
        "status_endpoint_anatomy_and_missingness_checks": {
            "covariate_branches": (
                "every nonempty frozen nuisance block plus all missingness indicators"
            ),
            "endpoint_branches": "every nonempty declared endpoint bin",
            "endpoint_operation": "pool into an explicit retained endpoint-control bin",
            "endpoint_mass_preserved": True,
            "common_models": list(MODEL_TRIPLET),
        },
    }
    payload: dict[str, Any] = {
        "schema_version": CONTROL_SPEC_SCHEMA,
        "purpose": "production implementation contract for seven EP12 controls",
        "control_pass_semantics": {
            "complete_prespecified_branch_set_executed": True,
            "applicability_checks_passed": True,
            "conservation_checks_passed": True,
            "capacity_checks_passed": True,
            "every_branch_has_complete_T_U_M_result": True,
            "robustness_or_residual_mode_support_decided_here": False,
            "new_scientific_pass_threshold": None,
        },
        "controls": controls,
        "development_roles_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    payload["spec_sha256"] = digest_object(payload)
    return payload


def validate_implementation_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = implementation_spec()
    if dict(payload) != expected:
        raise FalsifierControlError("control implementation spec is not frozen")
    return expected


def _array_digest(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    material = (
        str(array.dtype).encode("ascii")
        + b"\0"
        + json.dumps(array.shape).encode("ascii")
        + b"\0"
        + array.tobytes()
    )
    return hashlib.sha256(material).hexdigest()


def pooled_multinomial_noise(counts: Any, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Generate deterministic same-shape counts with every row total fixed."""

    import numpy as np

    source = np.asarray(counts)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.integer):
        raise FalsifierControlError("noise input must be a two-dimensional integer matrix")
    source = source.astype(np.int64, copy=False)
    if np.any(source < 0) or source.shape[1] < 2:
        raise FalsifierControlError("noise input is negative or has too few columns")
    row_totals = source.sum(axis=1)
    if np.any(row_totals <= 0):
        raise FalsifierControlError("noise input contains an empty row")
    pooled = source.sum(axis=0).astype(float)
    if float(pooled.sum()) <= 0:
        raise FalsifierControlError("noise input has no mass")
    probabilities = pooled / pooled.sum()
    rng = np.random.default_rng(int(seed))
    generated = np.vstack(
        [rng.multinomial(int(total), probabilities) for total in row_totals]
    ).astype(np.int64)
    row_ok = bool(np.array_equal(generated.sum(axis=1), row_totals))
    shape_ok = generated.shape == source.shape
    if not row_ok or not shape_ok or np.any(generated < 0):
        raise FalsifierControlError("pooled multinomial generator violated capacity")
    return generated, {
        "transform": "deterministic_pooled_multinomial_noise",
        "seed": int(seed),
        "input_sha256": _array_digest(source),
        "output_sha256": _array_digest(generated),
        "same_shape": shape_ok,
        "row_totals_preserved": row_ok,
        "pooled_probability_sha256": _array_digest(probabilities),
    }


def pooled_multinomial_noise_sparse(counts: Any, *, seed: int) -> tuple[Any, dict[str, Any]]:
    """Sparse production equivalent of :func:`pooled_multinomial_noise`.

    Only one generated row is dense at a time.  This keeps peak memory bounded
    by the vocabulary width instead of materializing a neuron-by-partner dense
    copy of the development snapshot.
    """

    import numpy as np
    from scipy import sparse

    source = sparse.csr_matrix(counts, dtype=np.int64)
    if source.ndim != 2 or source.shape[1] < 2:
        raise FalsifierControlError("noise input must have at least two columns")
    if source.nnz and np.any(source.data < 0):
        raise FalsifierControlError("noise input contains negative counts")
    row_totals = np.asarray(source.sum(axis=1)).ravel().astype(np.int64)
    if np.any(row_totals <= 0):
        raise FalsifierControlError("noise input contains an empty row")
    pooled = np.asarray(source.sum(axis=0)).ravel().astype(float)
    if float(pooled.sum()) <= 0:
        raise FalsifierControlError("noise input has no mass")
    probabilities = pooled / pooled.sum()
    rng = np.random.default_rng(int(seed))
    data: list[int] = []
    indices: list[int] = []
    indptr = [0]
    output_digest = hashlib.sha256()
    for total in row_totals:
        row = rng.multinomial(int(total), probabilities).astype(np.int64)
        output_digest.update(row.tobytes())
        nonzero = np.flatnonzero(row)
        indices.extend(int(value) for value in nonzero)
        data.extend(int(row[value]) for value in nonzero)
        indptr.append(len(data))
    generated = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=source.shape,
    )
    row_ok = bool(
        np.array_equal(
            np.asarray(generated.sum(axis=1)).ravel().astype(np.int64), row_totals
        )
    )
    if not row_ok or generated.shape != source.shape:
        raise FalsifierControlError("pooled multinomial generator violated capacity")
    return generated, {
        "transform": "deterministic_pooled_multinomial_noise",
        "seed": int(seed),
        "input_sha256": digest_object(
            {
                "shape": list(source.shape),
                "data": _array_digest(source.data),
                "indices": _array_digest(source.indices),
                "indptr": _array_digest(source.indptr),
            }
        ),
        "output_dense_row_stream_sha256": output_digest.hexdigest(),
        "same_shape": True,
        "row_totals_preserved": row_ok,
        "total_mass_preserved": int(generated.sum()) == int(source.sum()),
        "pooled_probability_sha256": _array_digest(probabilities),
    }


def nuisance_block_columns(*, spline_df: int, design_width: int) -> dict[str, list[int]]:
    """Map the frozen covariate construction to named nonempty blocks."""

    if spline_df not in {0, 3, 5}:
        raise FalsifierControlError("unsupported nuisance spline branch")
    expected = 22 + (0 if spline_df == 0 else 6 * (spline_df - 1))
    if design_width != expected:
        raise FalsifierControlError(
            f"covariate width {design_width} differs from frozen width {expected}"
        )
    spline_columns = list(range(22, expected))
    blocks = {
        "connection_strength_exposure": list(range(6, 10)),
        "reconstruction_and_status": [10] + list(range(17, 22)),
        "prespecified_anatomical_effects": (
            list(range(0, 6)) + list(range(11, 17)) + spline_columns
        ),
        "missingness_indicators": list(range(11, 22)),
    }
    if any(not values for values in blocks.values()):
        raise FalsifierControlError("a frozen nuisance block is empty")
    return blocks


def ablate_nuisance_block(
    covariates: Any,
    *,
    spline_df: int,
    block_id: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from .falsifier_transforms import ablate_covariate_columns

    source = np.asarray(covariates, dtype=float)
    blocks = nuisance_block_columns(
        spline_df=int(spline_df), design_width=int(source.shape[1])
    )
    if block_id not in blocks:
        raise FalsifierControlError(f"unknown nuisance block: {block_id}")
    result, receipt = ablate_covariate_columns(
        source, blocks[block_id], block_id=block_id
    )
    receipt = dict(receipt)
    receipt["input_sha256"] = _array_digest(source)
    receipt["output_sha256"] = _array_digest(result)
    receipt["all_declared_columns_zero"] = bool(
        np.all(result[:, blocks[block_id]] == 0.0)
    )
    return result, receipt


def endpoint_bin(raw_key: str) -> str:
    if raw_key == "missing_annotation":
        return "missing_annotation"
    if raw_key.startswith("typed::"):
        return "other_typed"
    if not raw_key.startswith("untyped_status::"):
        return "out_of_vocabulary"
    try:
        _, status, status_label = raw_key.split("::", 2)
    except ValueError as error:
        raise FalsifierControlError(f"malformed endpoint key: {raw_key}") from error
    normalized = f"{status} {status_label}".lower()
    if "orphan" in normalized or "leaves" in normalized:
        return "fragment"
    if status in {"Assign", "Anchor"} or any(
        token in normalized
        for token in ("hard to trace", "partially traced", "anchor", "assign")
    ):
        return "proofreading"
    if status == "Traced":
        return "untyped"
    return "unknown"


def endpoint_branch_inventory(
    raw_vocabulary: Sequence[str], counts: Any
) -> list[dict[str, Any]]:
    """Enumerate every nonempty declared endpoint bin without a mass cutoff."""

    import numpy as np

    matrix = np.asarray(counts)
    if matrix.ndim != 2 or matrix.shape[1] != len(raw_vocabulary):
        raise FalsifierControlError("endpoint inventory dimensions disagree")
    inventory: list[dict[str, Any]] = []
    for endpoint in ENDPOINT_BINS:
        columns = [
            index
            for index, key in enumerate(raw_vocabulary)
            if endpoint_bin(str(key)) == endpoint
        ]
        mass = int(matrix[:, columns].sum()) if columns else 0
        if mass > 0:
            inventory.append(
                {
                    "branch_id": f"endpoint::{endpoint}",
                    "endpoint_bin": endpoint,
                    "raw_columns": columns,
                    "observed_mass": mass,
                }
            )
    if not inventory:
        raise FalsifierControlError("no nonempty endpoint branch is available")
    return inventory


def partner_family_inventory(
    hierarchy: Mapping[str, Mapping[str, Any]],
    raw_vocabulary: Sequence[str],
) -> list[dict[str, Any]]:
    """Enumerate all nonempty subclass/class families represented in vocabulary."""

    typed = {
        str(key).removeprefix("typed::")
        for key in raw_vocabulary
        if str(key).startswith("typed::")
    }
    inventory: list[dict[str, Any]] = []
    for depth, field in ((1, "subclass"), (2, "class")):
        grouped: dict[str, list[str]] = {}
        for provider_type in sorted(typed):
            record = hierarchy.get(provider_type)
            if not isinstance(record, Mapping):
                raise FalsifierControlError(
                    f"missing hierarchy entry for typed partner {provider_type}"
                )
            label = str(record.get(field, "")).strip() or "unclassified"
            grouped.setdefault(label, []).append(f"typed::{provider_type}")
        for label, keys in sorted(grouped.items()):
            inventory.append(
                {
                    "branch_id": f"partner_family::depth{depth}::{label}",
                    "hierarchy_depth": depth,
                    "hierarchy_field": field,
                    "family_label": label,
                    "raw_partner_keys": keys,
                }
            )
    if not inventory:
        raise FalsifierControlError("no hierarchy-defined partner family is available")
    return inventory


def component_partner_contrasts(
    *,
    basis: Any,
    component_means: Any,
    component_weights: Any,
) -> dict[str, Any]:
    """Convert source-side M residual means to centered partner CLR contrasts."""

    import numpy as np

    basis_array = np.asarray(basis, dtype=float)
    means = np.asarray(component_means, dtype=float)
    weights = np.asarray(component_weights, dtype=float)
    if basis_array.ndim != 2 or means.ndim != 2:
        raise FalsifierControlError("component contrast arrays must be two-dimensional")
    if means.shape[1] != basis_array.shape[1] or weights.shape != (means.shape[0],):
        raise FalsifierControlError("component contrast dimensions disagree")
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise FalsifierControlError("component weights are invalid")
    partner_clr = means @ basis_array.T
    contrasts = partner_clr - weights @ partner_clr
    if not np.all(np.isfinite(contrasts)):
        raise FalsifierControlError("component partner contrasts are nonfinite")
    return {
        "component_count": int(means.shape[0]),
        "component_weights": weights.tolist(),
        "partner_contrasts": contrasts.tolist(),
        "partner_contrasts_sha256": _array_digest(contrasts),
        "weighted_component_mean_is_zero": bool(
            np.allclose(weights @ contrasts, 0.0, atol=1e-10)
        ),
    }


def label_invariant_component_score(left: Any, right: Any) -> dict[str, Any]:
    """Return the exact Hungarian-equivalent score for K in {2, 3}."""

    import numpy as np

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if (
        left_array.ndim != 2
        or right_array.shape != left_array.shape
        or left_array.shape[0] not in {2, 3}
    ):
        raise FalsifierControlError("alignment needs matching K=2 or K=3 contrasts")
    left_norm = np.linalg.norm(left_array, axis=1)
    right_norm = np.linalg.norm(right_array, axis=1)
    if np.any(left_norm <= 0) or np.any(right_norm <= 0):
        raise FalsifierControlError("alignment contrast has zero norm")
    cosine = (left_array @ right_array.T) / (left_norm[:, None] * right_norm[None, :])
    candidates = []
    for permutation in itertools.permutations(range(left_array.shape[0])):
        score = float(
            np.mean([cosine[index, permutation[index]] for index in range(len(permutation))])
        )
        candidates.append((score, tuple(int(value) for value in permutation)))
    score, permutation = max(candidates, key=lambda value: (value[0], tuple(-x for x in value[1])))
    return {
        "score": score,
        "component_permutation": list(permutation),
        "cosine_matrix": cosine.tolist(),
    }


def cross_side_alignment_permutation(
    records: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Any]:
    """Score unpermuted and deterministic within-K right-type pairings."""

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        left = record.get("left")
        right = record.get("right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise FalsifierControlError("alignment record lacks left/right fits")
        left_k = int(left.get("component_count", 0))
        right_k = int(right.get("component_count", 0))
        if left_k != right_k or left_k not in {2, 3}:
            raise FalsifierControlError("alignment component-count stratum is invalid")
        grouped.setdefault(left_k, []).append(record)
    if not grouped or all(len(values) < 2 for values in grouped.values()):
        raise FalsifierControlError("no component-count stratum supports a permutation")

    unpermuted: list[dict[str, Any]] = []
    permuted: list[dict[str, Any]] = []
    for component_count, stratum in sorted(grouped.items()):
        ordered = sorted(stratum, key=lambda value: str(value["provider_type"]))
        size = len(ordered)
        if size == 1:
            # A singleton remains fixed inside its stratum.  Applicability is
            # still established globally by the check above, which requires at
            # least one stratum with two or more provider types.
            shift = 0
            right_order = ordered
        else:
            material = f"{int(seed)}\0{component_count}\0{size}".encode("utf-8")
            shift = (
                int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
                % (size - 1)
                + 1
            )
            right_order = ordered[shift:] + ordered[:shift]
        for left_record, permuted_right in zip(ordered, right_order):
            provider_type = str(left_record["provider_type"])
            unpermuted_score = label_invariant_component_score(
                left_record["left"]["partner_contrasts"],
                left_record["right"]["partner_contrasts"],
            )
            permuted_score = label_invariant_component_score(
                left_record["left"]["partner_contrasts"],
                permuted_right["right"]["partner_contrasts"],
            )
            unpermuted.append(
                {
                    "provider_type": provider_type,
                    "right_provider_type": provider_type,
                    "component_count": component_count,
                    **unpermuted_score,
                }
            )
            permuted.append(
                {
                    "provider_type": provider_type,
                    "right_provider_type": str(permuted_right["provider_type"]),
                    "component_count": component_count,
                    "stratum_shift": shift,
                    **permuted_score,
                }
            )
    branches = [
        {
            "branch_id": "unpermuted",
            "pair_scores": unpermuted,
            "pair_score_sha256": digest_object(unpermuted),
        },
        {
            "branch_id": "deterministic_within_K_permutation",
            "pair_scores": permuted,
            "pair_score_sha256": digest_object(permuted),
        },
    ]
    return {
        "branches": branches,
        "branch_count": len(branches),
        "all_pairings_label_invariant": True,
        "right_pairing_permuted_only_within_component_count": True,
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise FalsifierControlError("cannot average an empty influence branch")
    return float(math.fsum(values) / len(values))


def leave_one_type_influence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the complete equal-type/equal-direction analytic deletion curve."""

    normalized: list[dict[str, Any]] = []
    for record in records:
        provider_type = str(record["provider_type"])
        left_scores = record.get("left_to_right_scores")
        right_scores = record.get("right_to_left_scores")
        if not isinstance(left_scores, Mapping) or not isinstance(right_scores, Mapping):
            raise FalsifierControlError("type influence needs complete directional scores")
        normalized.append(
            {
                "provider_type": provider_type,
                "left_to_right": {
                    **{name: float(left_scores[name]) for name in ("T", "U", "M")},
                    "delta": float(record["left_to_right_delta"]),
                },
                "right_to_left": {
                    **{name: float(right_scores[name]) for name in ("T", "U", "M")},
                    "delta": float(record["right_to_left_delta"]),
                },
            }
        )
    if len(normalized) < 2:
        raise FalsifierControlError("leave-one-type curve needs at least two complete types")

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {
            metric: _mean(
                row[direction][metric]
                for row in rows
                for direction in ("left_to_right", "right_to_left")
            )
            for metric in ("T", "U", "M", "delta")
        }

    full = aggregate(normalized)
    curve = []
    for omitted in sorted(row["provider_type"] for row in normalized):
        retained = [row for row in normalized if row["provider_type"] != omitted]
        estimate = aggregate(retained)
        curve.append(
            {
                "omitted_provider_type": omitted,
                "remaining_type_count": len(retained),
                "aggregate": estimate,
                "change_from_full": {
                    key: estimate[key] - full[key] for key in full
                },
            }
        )
    return {
        "full_equal_type_direction_aggregate": full,
        "leave_one_type_curve": curve,
        "complete_type_count": len(normalized),
        "curve_complete": len(curve) == len(normalized),
    }


def neuron_influence_curves(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute all single-neuron and full high-strength deletion curves."""

    metrics = ("T", "U", "M", "delta")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        provider_type = str(record["provider_type"])
        directional = record.get("target_neuron_scores")
        if not isinstance(directional, Mapping):
            raise FalsifierControlError("neuron influence needs target-neuron scores")
        for direction in ("left_to_right", "right_to_left"):
            values = directional.get(direction)
            if not isinstance(values, list) or not values:
                raise FalsifierControlError("a type/direction has no target-neuron scores")
            normalized = []
            for value in values:
                if not isinstance(value, Mapping):
                    raise FalsifierControlError("target-neuron score is malformed")
                normalized.append(
                    {
                        "body_id": str(value["body_id"]),
                        "strength": int(value["strength"]),
                        **{metric: float(value[metric]) for metric in metrics},
                    }
                )
            cells[(provider_type, direction)] = normalized
    if not cells or any(len(values) < 2 for values in cells.values()):
        raise FalsifierControlError(
            "every type/direction needs at least two neurons for full influence curves"
        )

    def aggregate(active: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict[str, float]:
        per_cell = {
            key: {metric: _mean(row[metric] for row in rows) for metric in metrics}
            for key, rows in active.items()
        }
        return {
            metric: _mean(value[metric] for value in per_cell.values())
            for metric in metrics
        }

    full = aggregate(cells)
    leave_one = []
    for key in sorted(cells):
        for neuron in sorted(cells[key], key=lambda value: value["body_id"]):
            active = {cell: list(values) for cell, values in cells.items()}
            active[key] = [
                value for value in active[key] if value["body_id"] != neuron["body_id"]
            ]
            estimate = aggregate(active)
            leave_one.append(
                {
                    "provider_type": key[0],
                    "direction": key[1],
                    "body_id": neuron["body_id"],
                    "strength": neuron["strength"],
                    "aggregate": estimate,
                    "change_from_full": {
                        metric: estimate[metric] - full[metric] for metric in metrics
                    },
                }
            )

    active = {cell: list(values) for cell, values in cells.items()}
    ordered = sorted(
        (
            (int(value["strength"]), key[0], key[1], str(value["body_id"]))
            for key, values in cells.items()
            for value in values
        ),
        key=lambda value: (-value[0], value[1], value[2], value[3]),
    )
    strength_curve = []
    for strength, provider_type, direction, body_id in ordered:
        key = (provider_type, direction)
        if len(active[key]) <= 1:
            continue
        active[key] = [value for value in active[key] if value["body_id"] != body_id]
        strength_curve.append(
            {
                "removed_rank": len(strength_curve) + 1,
                "provider_type": provider_type,
                "direction": direction,
                "body_id": body_id,
                "strength": strength,
                "remaining_neuron_count": sum(len(values) for values in active.values()),
                "aggregate": aggregate(active),
            }
        )
    expected_single = sum(len(values) for values in cells.values())
    expected_strength = sum(len(values) - 1 for values in cells.values())
    return {
        "full_equal_type_direction_aggregate": full,
        "leave_one_neuron_curve": leave_one,
        "high_strength_cumulative_curve": strength_curve,
        "leave_one_neuron_curve_complete": len(leave_one) == expected_single,
        "high_strength_curve_complete": len(strength_curve) == expected_strength,
        "new_concentration_cutoff": None,
    }


def make_branch_manifest(
    *,
    control_id: str,
    branch_id: str,
    branch_kind: str,
    complete_triplet_sha256: str,
    applicability_checks_passed: bool,
    conservation_checks_passed: bool,
    capacity_checks_passed: bool,
    artifacts: Mapping[str, str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if control_id not in COUNTABLE_COMPOSITE_CONTROLS:
        raise FalsifierControlError(f"unknown composite control: {control_id}")
    if not branch_id or not complete_triplet_sha256:
        raise FalsifierControlError("branch identity and T/U/M hash are required")
    manifest: dict[str, Any] = {
        "schema_version": BRANCH_MANIFEST_SCHEMA,
        "control_id": control_id,
        "branch_id": branch_id,
        "branch_kind": branch_kind,
        "complete_T_U_M_result_sha256": complete_triplet_sha256,
        "complete_T_U_M_result": True,
        "applicability_checks_passed": bool(applicability_checks_passed),
        "conservation_checks_passed": bool(conservation_checks_passed),
        "capacity_checks_passed": bool(capacity_checks_passed),
        "artifacts": dict(sorted((str(k), str(v)) for k, v in artifacts.items())),
        "details": dict(details or {}),
        "development_roles_only": True,
        "final_connectivity_accessed": False,
    }
    manifest["branch_sha256"] = digest_object(manifest)
    return manifest


def composite_control_receipt(
    *,
    control_id: str,
    expected_branch_ids: Sequence[str],
    branches: Sequence[Mapping[str, Any]],
    implementation_spec_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless the complete branch set and every check pass."""

    if control_id not in COUNTABLE_COMPOSITE_CONTROLS:
        raise FalsifierControlError(f"unknown composite control: {control_id}")
    expected = list(map(str, expected_branch_ids))
    observed = [str(branch.get("branch_id")) for branch in branches]
    complete_set = len(expected) == len(set(expected)) and observed == expected
    checks = [
        bool(branch.get("complete_T_U_M_result") is True)
        and bool(branch.get("applicability_checks_passed") is True)
        and bool(branch.get("conservation_checks_passed") is True)
        and bool(branch.get("capacity_checks_passed") is True)
        for branch in branches
    ]
    hashes_valid = all(
        digest_object({k: v for k, v in branch.items() if k != "branch_sha256"})
        == branch.get("branch_sha256")
        for branch in branches
    )
    passed = bool(complete_set and branches and all(checks) and hashes_valid)
    receipt: dict[str, Any] = {
        "schema_version": CONTROL_RECEIPT_SCHEMA,
        "falsifier_id": control_id,
        "implementation_spec_sha256": implementation_spec_sha256,
        "expected_branch_ids": expected,
        "observed_branch_ids": observed,
        "branch_manifests": [dict(value) for value in branches],
        "complete_prespecified_branch_set_executed": complete_set,
        "applicability_checks_passed": bool(branches and all(checks)),
        "conservation_checks_passed": bool(
            branches and all(branch.get("conservation_checks_passed") is True for branch in branches)
        ),
        "capacity_checks_passed": bool(
            branches and all(branch.get("capacity_checks_passed") is True for branch in branches)
        ),
        "every_branch_complete_T_U_M": bool(
            branches and all(branch.get("complete_T_U_M_result") is True for branch in branches)
        ),
        "branch_hashes_valid": hashes_valid,
        "control_passed": passed,
        "robustness_or_support_decided": False,
        "new_scientific_pass_threshold": None,
        "development_roles_only": True,
        "final_connectivity_accessed": False,
    }
    receipt["receipt_sha256"] = digest_object(receipt)
    if not passed:
        raise FalsifierControlError("composite control branch set is incomplete or invalid")
    return receipt
