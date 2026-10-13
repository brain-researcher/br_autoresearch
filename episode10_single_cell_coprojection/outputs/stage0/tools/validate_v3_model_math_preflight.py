#!/usr/bin/env python3
"""Target-free mathematical preflight for the EP10 v3 fixed-K model.

Only deterministic synthetic observations are used.  This program neither
accepts nor opens an EP10 target-call table, morphology, or role ledger.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def subsets(p: int, k: int) -> tuple[tuple[int, ...], ...]:
    if not 0 <= k <= p:
        raise ValueError("K must be between zero and p")
    return tuple(itertools.combinations(range(p), k))


def feature_matrix(p: int, k: int, include_pairs: bool = True) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    states = subsets(p, k)
    pairs = tuple(itertools.combinations(range(p), 2)) if include_pairs else ()
    matrix = np.zeros((len(states), p + len(pairs)), dtype=float)
    for row_index, state in enumerate(states):
        present = set(state)
        matrix[row_index, list(state)] = 1.0
        for pair_index, (left, right) in enumerate(pairs, start=p):
            matrix[row_index, pair_index] = float(left in present and right in present)
    return states, matrix


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def nll_and_gradient(theta: np.ndarray, features: np.ndarray, observed_index: int) -> tuple[float, np.ndarray]:
    energies = features @ theta
    log_partition = logsumexp(energies)
    probabilities = np.exp(energies - log_partition)
    nll = log_partition - float(energies[observed_index])
    gradient = probabilities @ features - features[observed_index]
    return nll, np.asarray(gradient, dtype=float)


def elementary_log_partition(singleton_theta: np.ndarray, k: int) -> float:
    coefficients = np.zeros(k + 1, dtype=np.longdouble)
    coefficients[0] = 1.0
    for weight in np.exp(singleton_theta).astype(np.longdouble):
        for degree in range(k, 0, -1):
            coefficients[degree] += weight * coefficients[degree - 1]
    return math.log(float(coefficients[k]))


def fit_singleton_conditional(observations: Sequence[tuple[int, ...]], p: int) -> dict[str, float]:
    # Fix the final singleton coefficient to zero to remove the conditional-K
    # additive-shift non-identifiability.
    caches: dict[int, tuple[tuple[tuple[int, ...], ...], np.ndarray, dict[tuple[int, ...], int]]] = {}
    for k in sorted({len(state) for state in observations}):
        states, features = feature_matrix(p, k, include_pairs=False)
        caches[k] = (states, features[:, : p - 1], {state: index for index, state in enumerate(states)})

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros_like(beta)
        for state in observations:
            _, features, indexes = caches[len(state)]
            value, grad = nll_and_gradient(beta, features, indexes[state])
            loss += value
            gradient += grad
        scale = 1.0 / len(observations)
        return loss * scale, gradient * scale

    beta = np.zeros(p - 1, dtype=float)
    initial, _ = objective(beta)
    accepted_steps = 0
    for _ in range(1200):
        value, gradient = objective(beta)
        norm = float(np.linalg.norm(gradient))
        if norm < 2.0e-7:
            break
        step = 1.0
        directional = norm * norm
        while step > 1.0e-10:
            candidate = beta - step * gradient
            candidate_value, _ = objective(candidate)
            if candidate_value <= value - 1.0e-4 * step * directional:
                beta = candidate
                accepted_steps += 1
                break
            step *= 0.5
        else:
            raise AssertionError("deterministic optimizer line search stalled")
    final, final_gradient = objective(beta)
    return {
        "initial_nll": initial,
        "final_nll": final,
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "accepted_steps": float(accepted_steps),
    }


def equal_group_score(values: Sequence[float], groups: Sequence[str]) -> float:
    if len(values) != len(groups) or not values:
        raise ValueError("values/groups must be nonempty and aligned")
    by_group: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    return float(np.mean([np.mean(group_values) for group_values in by_group.values()]))


def constrained_permutation(records: Sequence[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    output = [dict(record) for record in records]
    strata: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        state = tuple(record["state"])
        key = (str(record["group"]), int(record["segment_unmapped"]), len(state))
        strata[key].append(index)
    generator = np.random.default_rng(seed)
    for indexes in strata.values():
        labels = [tuple(records[index]["state"]) for index in indexes]
        generator.shuffle(labels)
        for index, state in zip(indexes, labels):
            output[index]["state"] = state
    return output


def _stratum_multisets(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int, int], tuple[tuple[int, ...], ...]]:
    collected: dict[tuple[str, int, int], list[tuple[int, ...]]] = defaultdict(list)
    for record in records:
        state = tuple(record["state"])
        key = (str(record["group"]), int(record["segment_unmapped"]), len(state))
        collected[key].append(state)
    return {key: tuple(sorted(values)) for key, values in collected.items()}


def run_preflight() -> dict[str, Any]:
    checks: list[str] = []

    # Exact p=16 normalization agrees with an independent elementary-symmetric
    # polynomial calculation for a singleton-only conditional model.
    theta16 = np.linspace(-0.7, 0.8, 16)
    states16, features16 = feature_matrix(16, 7, include_pairs=False)
    enumerated = logsumexp(features16 @ theta16)
    independent = elementary_log_partition(theta16, 7)
    if len(states16) != math.comb(16, 7) or not math.isclose(enumerated, independent, rel_tol=0.0, abs_tol=2.0e-12):
        raise AssertionError("exact p=16 fixed-K partition check failed")
    checks.append("exact_fixed_K_log_partition_p16")

    # General singleton+pair gradient versus central finite differences.
    states8, features8 = feature_matrix(8, 3, include_pairs=True)
    theta8 = np.linspace(-0.25, 0.3, features8.shape[1])
    observed_index = states8.index((1, 4, 7))
    _, analytic = nll_and_gradient(theta8, features8, observed_index)
    finite = np.zeros_like(theta8)
    epsilon = 1.0e-6
    for index in range(theta8.size):
        plus = theta8.copy()
        minus = theta8.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite[index] = (
            nll_and_gradient(plus, features8, observed_index)[0]
            - nll_and_gradient(minus, features8, observed_index)[0]
        ) / (2.0 * epsilon)
    max_gradient_error = float(np.max(np.abs(analytic - finite)))
    if max_gradient_error > 2.0e-8:
        raise AssertionError(f"analytic gradient error {max_gradient_error}")
    checks.append("analytic_gradient_matches_finite_difference")

    # Deterministic, nonseparable synthetic counts exercise fitting without a
    # third-party optimizer.
    p_fit = 6
    fit_states, fit_features = feature_matrix(p_fit, 2, include_pairs=False)
    true_theta = np.asarray((0.9, 0.45, 0.1, -0.2, -0.55, 0.0))
    probabilities = np.exp(fit_features @ true_theta - logsumexp(fit_features @ true_theta))
    counts = np.maximum(1, np.rint(800.0 * probabilities).astype(int))
    observations = [state for state, count in zip(fit_states, counts) for _ in range(int(count))]
    optimizer = fit_singleton_conditional(observations, p_fit)
    if (
        optimizer["final_nll"] >= optimizer["initial_nll"] - 0.01
        or optimizer["final_gradient_norm"] > 2.0e-6
        or optimizer["accepted_steps"] < 1
    ):
        raise AssertionError(f"optimizer smoke failed: {optimizer}")
    checks.append("deterministic_optimizer_smoke")

    base_score = equal_group_score((1.0, 3.0, 10.0), ("A", "A", "B"))
    duplicated_score = equal_group_score((1.0, 3.0, 1.0, 3.0, 10.0), ("A", "A", "A", "A", "B"))
    if base_score != 6.0 or duplicated_score != base_score:
        raise AssertionError("equal-group scoring is not invariant to within-group replication")
    checks.append("equal_group_weighted_score")

    records = [
        {"group": "g1", "segment_unmapped": 0, "state": (0, 1)},
        {"group": "g1", "segment_unmapped": 0, "state": (2, 3)},
        {"group": "g1", "segment_unmapped": 1, "state": (1, 4)},
        {"group": "g1", "segment_unmapped": 1, "state": (0, 5)},
        {"group": "g2", "segment_unmapped": 0, "state": ()},
        {"group": "g2", "segment_unmapped": 0, "state": ()},
        {"group": "g2", "segment_unmapped": 1, "state": tuple(range(16))},
        {"group": "g2", "segment_unmapped": 1, "state": tuple(range(16))},
    ]
    permuted = constrained_permutation(records, seed=20260926)
    if _stratum_multisets(permuted) != _stratum_multisets(records):
        raise AssertionError("constrained permutation changed group/mapping/K strata")
    if permuted != constrained_permutation(records, seed=20260926):
        raise AssertionError("constrained permutation is not deterministic")
    checks.append("group_mapping_stratum_K_permutation_invariance")

    for k in (0, 16):
        states, features = feature_matrix(16, k, include_pairs=True)
        theta = np.linspace(-0.1, 0.1, features.shape[1])
        nll, gradient = nll_and_gradient(theta, features, 0)
        if len(states) != 1 or abs(nll) > 1.0e-12 or float(np.max(np.abs(gradient))) > 1.0e-12:
            raise AssertionError(f"degenerate K={k} did not have exact zero gain")
    checks.append("degenerate_K_0_and_16_zero_gain")

    return {
        "status": "pass",
        "stage": "v3_target_free_model_math_preflight",
        "uses_real_targets_morphologies_or_roles": False,
        "p": 16,
        "exact_state_count_K7": len(states16),
        "max_gradient_absolute_error": max_gradient_error,
        "optimizer": optimizer,
        "checks": checks,
        "final_or_unassigned_opened": False,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run deterministic synthetic preflight")
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("synthetic execution requires --self-test")
    result = run_preflight()
    if args.output is not None:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
