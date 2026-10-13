"""Outcome-blind oracle primitives for EP12 generated counterexamples.

This module never imports or calls the fitted T/U/M implementation.  It can
evaluate the known generating distribution with target latent coordinates and
groups marginalized, and it can classify already-computed directional bounds
against the frozen margin.  It intentionally fails closed before defining a
population M-minus-best-T/U gain: the frozen files do not yet identify a
unique population projection for the bounded curved-U reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

try:
    import numpy as np
    from scipy.special import gammaln, logsumexp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("synthetic oracle primitives require NumPy and SciPy") from exc

from .scientific_synthetic import SyntheticSideDGP, generator_probabilities


Array = np.ndarray
GROUP_GENERATORS = frozenset(
    {"residual_groups_within_continuum", "weak_residual_groups"}
)


class OracleError(ValueError):
    """The requested generated-truth calculation is invalid."""


class OracleScientificChoiceRequired(OracleError):
    """A population reference is scientifically under-specified."""


@dataclass(frozen=True)
class OracleMarginalScore:
    per_neuron_log_predictive_density_per_count: Array
    quadrature_nodes: int
    latent_family: str
    target_latent_coordinate: str = "marginalized"
    target_group: str = "marginalized"
    fitted_qualification_results_used: bool = False


def _normal_quadrature(nodes: int) -> tuple[Array, Array]:
    if isinstance(nodes, bool) or int(nodes) != nodes or int(nodes) < 1:
        raise OracleError("quadrature_nodes must be a positive integer")
    locations, weights = np.polynomial.hermite_e.hermegauss(int(nodes))
    weights = weights / math.sqrt(2.0 * math.pi)
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 1.0):
        raise OracleError("normal quadrature weights are invalid")
    return np.asarray(locations, dtype=float), np.asarray(weights, dtype=float)


def _latent_support(
    state: SyntheticSideDGP, quadrature_nodes: int
) -> list[tuple[float, float | None, float]]:
    if state.latent_family in {"point_mass", "deterministic_offset"}:
        return [(0.0, None, 1.0)]

    normal_locations, normal_weights = _normal_quadrature(quadrature_nodes)
    if state.latent_family in {"normal_linear", "normal_curved"}:
        return [
            (float(location), None, float(weight))
            for location, weight in zip(normal_locations, normal_weights)
        ]
    if state.latent_family == "sparse_curved":
        return [
            (
                float(1.45 * pole + 0.22 * location),
                None,
                float(0.5 * weight),
            )
            for pole in (-1.0, 1.0)
            for location, weight in zip(normal_locations, normal_weights)
        ]
    if state.latent_family == "grouped_normal_linear":
        if (
            len(state.group_centers) != state.group_components
            or len(state.group_weights) != state.group_components
            or np.any(state.group_weights <= 0.0)
            or not np.isclose(state.group_weights.sum(), 1.0)
        ):
            raise OracleError("grouped generator state has invalid component weights")
        return [
            (
                float(location),
                float(center),
                float(group_weight * normal_weight),
            )
            for center, group_weight in zip(
                state.group_centers, state.group_weights
            )
            for location, normal_weight in zip(normal_locations, normal_weights)
        ]
    raise OracleError(f"unknown synthetic latent family: {state.latent_family}")


def true_generator_log_predictive_density(
    counts: Array,
    state: SyntheticSideDGP,
    *,
    quadrature_nodes: int,
) -> OracleMarginalScore:
    """Score count vectors under their known marginal generator distribution.

    The caller supplies the quadrature size because no oracle-specific
    integration setting is frozen.  This function reports the approximation;
    it does not promote it into a margin-relative population truth claim.
    """

    values = np.asarray(counts)
    if values.ndim != 2 or values.shape != state.realized_probabilities.shape:
        raise OracleError("counts and synthetic DGP state have incompatible shapes")
    if not np.issubdtype(values.dtype, np.integer):
        if not np.all(np.equal(values, np.round(values))):
            raise OracleError("oracle counts must be integer-valued")
        values = np.round(values).astype(np.int64)
    else:
        values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise OracleError("oracle counts cannot be negative")
    totals = values.sum(axis=1)
    if not np.array_equal(totals, state.count_totals):
        raise OracleError("oracle count totals differ from the generated design")

    support = _latent_support(state, quadrature_nodes)
    result = np.empty(values.shape[0], dtype=float)
    for neuron_index, (row, total) in enumerate(zip(values, totals)):
        coefficient = float(
            gammaln(float(total) + 1.0)
            - np.sum(gammaln(row.astype(float) + 1.0))
        )
        terms: list[float] = []
        for latent_coordinate, group_center, weight in support:
            probabilities = generator_probabilities(
                state,
                neuron_index,
                latent_coordinate=latent_coordinate,
                group_center=group_center,
            )
            terms.append(
                math.log(weight)
                + coefficient
                + float(np.log(probabilities) @ row.astype(float))
            )
        result[neuron_index] = float(logsumexp(terms)) / float(total)
    if not np.all(np.isfinite(result)):
        raise OracleError("oracle marginal score is non-finite")
    return OracleMarginalScore(
        per_neuron_log_predictive_density_per_count=result,
        quadrature_nodes=int(quadrature_nodes),
        latent_family=state.latent_family,
    )


def classify_directional_margin_interval(
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    margin: float,
) -> str:
    """Classify two directional bounds without adding a new threshold."""

    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if lower_values.shape != (2,) or upper_values.shape != (2,):
        raise OracleError("margin classification requires exactly two directions")
    if (
        not np.all(np.isfinite(lower_values))
        or not np.all(np.isfinite(upper_values))
        or not math.isfinite(float(margin))
        or np.any(lower_values > upper_values)
    ):
        raise OracleError("margin classification bounds are invalid")
    if np.all(lower_values > float(margin)):
        return "above_margin"
    if np.all(upper_values < float(margin)):
        return "below_margin"
    return "margin_indeterminate"


def population_kl_group_gain(*, state: SyntheticSideDGP) -> None:
    """Fail closed until the population best-T/U projection is frozen.

    The exact M law is available above.  What is missing is not an engineering
    tolerance: EP12 has not frozen whether its oracle U is the global
    population-KL projection of the declared family, the population limit of
    the finite-sample ridge/CV algorithm, or another source-side projection.
    These alternatives can change which cells cross 0.02.
    """

    if state.scenario not in GROUP_GENERATORS:
        raise OracleError("population group gain requires a group generator")
    raise OracleScientificChoiceRequired(
        "population M-minus-best-T/U truth is undefined: freeze the source-side "
        "population T/U projection and U-curved optimization before calibration"
    )
