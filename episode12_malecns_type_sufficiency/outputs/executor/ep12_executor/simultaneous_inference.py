"""Shared whole-type paired, studentized simultaneous inference for EP12.

Rows are the uncertainty units.  A bootstrap draw resamples each row once and
uses the same indices for every direction and diagnostic coordinate.  The
bootstrap statistic is genuinely studentized: every resample recomputes its
own standard error before the maximum absolute t statistic is formed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SimultaneousInferenceError(ValueError):
    """Raised when a simultaneous interval cannot be computed faithfully."""


def bilateral_with_aggregate(values: Any) -> np.ndarray:
    """Return paired left-to-right, right-to-left, and mean-direction columns."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise SimultaneousInferenceError(
            "bilateral values must have shape (whole_type, 2)"
        )
    if array.shape[0] < 2 or not np.all(np.isfinite(array)):
        raise SimultaneousInferenceError(
            "bilateral values need at least two finite whole-type rows"
        )
    return np.column_stack((array, array.mean(axis=1)))


def paired_studentized_max_t(
    values: Any,
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute paired simultaneous intervals with a bootstrap-t maximum.

    Constant coordinates remain point intervals and do not enter the maximum.
    A nonconstant coordinate whose resampled standard error degenerates is
    represented by an infinite bootstrap statistic.  If that makes the
    requested critical value nonfinite, inference fails closed rather than
    silently clipping the statistic.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise SimultaneousInferenceError(
            "simultaneous inference needs >=2 rows and >=1 coordinate"
        )
    if not np.all(np.isfinite(array)):
        raise SimultaneousInferenceError("simultaneous inputs must be finite")
    if isinstance(replicates, bool) or int(replicates) != replicates or replicates < 1:
        raise SimultaneousInferenceError("bootstrap replicates must be a positive integer")
    if not 0.0 < float(confidence_level) < 1.0:
        raise SimultaneousInferenceError("confidence level must lie strictly in (0, 1)")

    n_rows, n_coordinates = array.shape
    estimates = array.mean(axis=0)
    standard_errors = array.std(axis=0, ddof=1) / math.sqrt(n_rows)
    variable = standard_errors > 0.0

    rng = np.random.Generator(np.random.PCG64(int(seed)))
    max_t = np.zeros(int(replicates), dtype=np.float64)
    zero_se_draws = 0

    # Bound both the sampled cube and its index matrix.  Drawing each batch
    # from the same PCG64 stream preserves the exact row-major sequence that a
    # single full allocation would have produced, while making large oracle
    # banks feasible.
    batch_size = max(
        1,
        min(
            int(replicates),
            5_000_000 // max(1, n_rows * n_coordinates),
            10_000_000 // max(1, n_rows),
        ),
    )
    for start in range(0, int(replicates), batch_size):
        stop = min(start + batch_size, int(replicates))
        indices = rng.integers(0, n_rows, size=(stop - start, n_rows))
        sampled = array[indices]
        bootstrap_means = sampled.mean(axis=1)
        if not np.any(variable):
            continue
        bootstrap_se = sampled.std(axis=1, ddof=1) / math.sqrt(n_rows)
        numerator = np.abs(bootstrap_means[:, variable] - estimates[variable])
        denominator = bootstrap_se[:, variable]
        zero = denominator <= 0.0
        zero_se_draws += int(np.count_nonzero(np.any(zero, axis=1)))
        statistic = np.empty_like(numerator)
        np.divide(numerator, denominator, out=statistic, where=~zero)
        # Exact equality is the only defensible zero-SE finite case.  Any
        # nonzero displacement with zero estimated uncertainty is infinite.
        statistic[zero & (numerator == 0.0)] = 0.0
        statistic[zero & (numerator != 0.0)] = np.inf
        max_t[start:stop] = statistic.max(axis=1)

    order = int(math.ceil(float(confidence_level) * int(replicates))) - 1
    critical = float(np.sort(max_t)[max(0, min(order, int(replicates) - 1))])
    if not math.isfinite(critical):
        raise SimultaneousInferenceError(
            "studentized bootstrap produced a nonfinite simultaneous critical value"
        )

    lower = estimates - critical * standard_errors
    upper = estimates + critical * standard_errors
    return {
        "method": "paired_whole_type_studentized_bootstrap_max_t",
        "confidence_level": float(confidence_level),
        "estimate": estimates.tolist(),
        "standard_error": standard_errors.tolist(),
        "critical_max_t": critical,
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "bootstrap_replicates": int(replicates),
        "bootstrap_zero_standard_error_draws": zero_se_draws,
        "variable_coordinate_count": int(np.count_nonzero(variable)),
        "directions_resampled_together": True,
        "unit": "whole_provider_type",
    }


__all__ = [
    "SimultaneousInferenceError",
    "bilateral_with_aggregate",
    "paired_studentized_max_t",
]
