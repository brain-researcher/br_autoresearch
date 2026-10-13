"""Pure training-array EP08 selectors and nested ridge fitting.

No filesystem, source-reader, evaluation-array or baseline-result interface.
The caller supplies only the currently allowed training rows and relative folds.
"""
from dataclasses import dataclass
import random
import time

import numpy as np

from ridge_training import ALPHAS, FittedRidge, fit_candidate, ridge_coefficients, scaling

SELECTORS = ("direction_repeatability", "marginal_target", "diversity",
             "conditional_target", "permuted_conditional_target")


def feature_columns(electrodes):
    return (3 * np.asarray(electrodes)[:, None] + np.arange(3)).ravel()


def selector_order(x, y, directions, method, maximum, permutation_seed=None):
    """Nested electrode positions, with exact ties settled by input position."""
    x, y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    directions = np.asarray(directions)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) != len(directions)
            or x.shape[1] % 3 or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Selectors require aligned finite training arrays and three features per electrode")
    groups, n = x.shape[1] // 3, len(x)
    if method not in SELECTORS or not 0 < maximum <= groups:
        raise ValueError("Unknown selector or unsupported retained count")
    xm, xs, _ = scaling(x)
    ym, ys, _ = scaling(y)
    x, y = (x - xm) / xs, (y - ym) / ys
    if method == "direction_repeatability":
        levels, counts = np.unique(directions, return_counts=True)
        if not np.array_equal(levels, np.arange(8)) or len(set(counts)) != 1 or counts[0] < 2:
            raise ValueError("Repeatability requires eight balanced directions and at least two fitting repeats")
        means = np.stack([x[directions == d].mean(axis=0) for d in levels])
        between = counts[0] * np.sum((means - x.mean(axis=0)) ** 2, axis=0) / 7
        within = sum(np.sum((x[directions == d] - means[d]) ** 2, axis=0) for d in levels) / (n - 8)
        component = np.maximum((between - within) / counts[0], 0)
        reliability = np.divide(component, component + within, out=np.zeros_like(component),
                                where=(component + within) > 0)
        scores = reliability.reshape(groups, 3).mean(axis=1)
        return np.argsort(-scores, kind="stable")[:maximum]
    if method == "permuted_conditional_target":
        if permutation_seed is None:
            raise ValueError("Control requires the prospective whole-trial permutation seed")
        rows = list(range(n))
        rng = random.Random()
        rng.seed(permutation_seed, version=2)
        rng.shuffle(rows)
        y = y[rows]  # One whole multivariate target row, never individual units/bins.
    # Maintain AX and AY for A=(I+X_S X_S'/n)^-1 by exact rank-three updates.
    # Candidate block normal matrices and target cross-products are only 3x3/3xU.
    ax, ay = x.copy(), y.copy()
    blocks = x.reshape(n, groups, 3)
    chosen, available = [], np.ones(groups, dtype=bool)
    for _ in range(1 if method == "marginal_target" else maximum):
        gram = np.einsum("ngb,ngc->gbc", blocks, ax.reshape(n, groups, 3)) / n
        gram = (gram + gram.transpose(0, 2, 1)) / 2
        matrix = gram + np.eye(3)[None, :, :]
        if method == "diversity":
            signs, scores = np.linalg.slogdet(matrix)
            if np.any(signs <= 0):
                raise ArithmeticError("Nonpositive block diversity matrix")
        else:
            cross = np.einsum("ngb,nu->gbu", blocks, ay) / n
            solutions = np.linalg.solve(matrix, cross)
            scores = np.sum(cross * solutions, axis=(1, 2)) / y.shape[1]
        if not np.isfinite(scores).all():
            raise ArithmeticError("Nonfinite selection score")
        if method == "marginal_target":
            return np.argsort(-scores, kind="stable")[:maximum]
        electrode = int(np.argmax(np.where(available, scores, -np.inf)))
        chosen.append(electrode)
        available[electrode] = False
        columns = feature_columns([electrode])
        update = ax[:, columns].copy()
        # Both right-hand sides use the pre-update state.
        rhs_x, rhs_y = x[:, columns].T @ ax / n, x[:, columns].T @ ay / n
        ax -= update @ np.linalg.solve(matrix[electrode], rhs_x)
        ay -= update @ np.linalg.solve(matrix[electrode], rhs_y)
    return np.asarray(chosen)


def validate_folds(n, folds):
    if len(folds) != 4:
        raise ValueError("Preserve four inner folds")
    seen = []
    for fit, valid in folds:
        f, v = list(fit), list(valid)
        if (not f or not v or len(f) != len(set(f)) or len(v) != len(set(v))
                or set(f) & set(v) or set(f) | set(v) != set(range(n))):
            raise ValueError("Inner fold must partition allowed training rows")
        seen.extend(v)
    if sorted(seen) != list(range(n)):
        raise ValueError("Inner validation rows must occur exactly once")


def build_order_bank(x, y, directions, folds, method, maximum, seed, progress):
    """Every inner selector is fitted before its own validation rows enter CV."""
    started = time.monotonic()
    inner = []
    for fold, (fit, _valid) in enumerate(folds):
        inner.append(selector_order(x[fit], y[fit], directions[fit], method, maximum,
                                    f"{seed}|{fold}"))
        progress["selector_fits"] += 1
    outer = selector_order(x, y, directions, method, maximum, f"{seed}|outer")
    progress["selector_fits"] += 1
    progress["selection_seconds"] += time.monotonic() - started
    return inner, outer


def fit_with_orders(x, y, folds, inner_orders, outer_order, count, progress):
    errors = np.zeros(len(ALPHAS))
    for (fit, valid), order in zip(folds, inner_orders):
        columns = feature_columns(order[:count])
        xf, xv, yf, yv = x[np.ix_(fit, columns)], x[np.ix_(valid, columns)], y[fit], y[valid]
        xm, xs, xc = scaling(xf)
        ym, ys, yc = scaling(yf)
        progress["zero_SD_input_column_fit_occurrences"] += len(xc)
        progress["zero_SD_target_column_fit_occurrences"] += len(yc)
        xf, xv, yf, yv = (xf-xm)/xs, (xv-xm)/xs, (yf-ym)/ys, (yv-ym)/ys
        for index, alpha in enumerate(ALPHAS):
            coefficients = ridge_coefficients(xf, yf, alpha)
            progress["inner_ridge_fits"] += 1
            error = float(np.sum((yv - xv @ coefficients) ** 2))
            if not np.isfinite(error):
                raise ArithmeticError("Nonfinite nested validation error")
            errors[index] += error
    best = [i for i, value in enumerate(errors) if value == errors.min()][-1]
    columns = feature_columns(outer_order[:count])
    xm, xs, xc = scaling(x[:, columns])
    ym, ys, yc = scaling(y)
    progress["zero_SD_input_column_fit_occurrences"] += len(xc)
    progress["zero_SD_target_column_fit_occurrences"] += len(yc)
    coefficients = ridge_coefficients((x[:, columns]-xm)/xs, (y-ym)/ys, ALPHAS[best])
    progress["outer_training_refits"] += 1
    return FittedRidge(xm, xs, ym, ys, coefficients, ALPHAS[best], xc, yc), columns


@dataclass(frozen=True)
class SelectedModel:
    selector: str
    electrodes: int
    replicate: int
    columns: np.ndarray
    fitted: FittedRidge


def fit_selection_panels(x, y, directions, folds, seed_prefix, progress,
                         counts=(8, 16, 32, 96), control_replicates=5):
    x, y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    directions = np.asarray(directions).copy()
    validate_folds(len(x), folds)
    groups = x.shape[1] // 3
    if counts[-1] != groups or any(c <= 0 or c > groups for c in counts):
        raise ValueError("Complete full-electrode comparator required")
    full = fit_candidate(x, y, folds, progress)
    results = []
    for method in SELECTORS:
        for replicate in range(control_replicates if method == "permuted_conditional_target" else 1):
            seed = f"{seed_prefix}|{replicate}"
            inner, outer = build_order_bank(x, y, directions, folds, method, max(counts[:-1]), seed, progress)
            for count in counts[:-1]:
                fitted, columns = fit_with_orders(x, y, folds, inner, outer, count, progress)
                results.append(SelectedModel(method, count, replicate, columns, fitted))
        # Shared identity: one full-electrode fit for all five panels.
        results.append(SelectedModel(method, groups, 0, np.arange(x.shape[1]), full))
    return results
