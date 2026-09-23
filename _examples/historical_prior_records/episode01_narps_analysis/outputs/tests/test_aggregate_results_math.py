#!/usr/bin/env python3
"""Black-box synthetic math audit of outputs/code/aggregate_results.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "outputs" / "code"))
import aggregate_results as production  # noqa: E402


def reference_ss_internal_csq(y_csqv: np.ndarray) -> dict[str, float]:
    # Convert the production C,S,Q array to the contract's C,Q,S notation.
    y = y_csqv.transpose(0, 2, 1, 3)
    grand = y.mean(axis=(0, 1, 2))
    ec = y.mean(axis=(1, 2)) - grand
    eq = y.mean(axis=(0, 2)) - grand
    es = y.mean(axis=(0, 1)) - grand
    ecq = (y.mean(2) - y.mean((1, 2))[:, None]
           - y.mean((0, 2))[None] + grand)
    ecs = (y.mean(1) - y.mean((1, 2))[:, None]
           - y.mean((0, 1))[None] + grand)
    eqs = (y.mean(0) - y.mean((0, 2))[:, None]
           - y.mean((0, 1))[None] + grand)
    ecqs = (y - grand - ec[:, None, None] - eq[None, :, None]
            - es[None, None] - ecq[:, :, None] - ecs[:, None]
            - eqs[None])
    return {
        "C": 9 * np.square(ec).sum(),
        "Q": 9 * np.square(eq).sum(),
        "S": 9 * np.square(es).sum(),
        "C:Q": 3 * np.square(ecq).sum(),
        "C:S": 3 * np.square(ecs).sum(),
        "Q:S": 3 * np.square(eqs).sum(),
        "C:Q:S": np.square(ecqs).sum(),
    }


def equal_cohort_weights(groups: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    w = np.zeros(len(groups), dtype=float)
    for group in (0, 1):
        use = eligible & (groups == group)
        w[use] = 0.5 / use.sum()
    return w


def test_axis_labels_and_projections(rng: np.random.Generator) -> None:
    for _ in range(25):
        y = rng.normal(size=(3, 3, 3, 29))  # production C,S,Q,V
        flat = y.reshape(27, 29)
        got = production.functional_anova(
            flat @ flat.T, production.projection_matrices()
        )["ss"]
        expected = reference_ss_internal_csq(y)
        for term in expected:
            np.testing.assert_allclose(got[term], expected[term], rtol=3e-13,
                                       atol=2e-12, err_msg=term)


def test_group_gram_order_and_bootstrap(rng: np.random.Generator) -> None:
    n, v, reps = 11, 37, 13
    maps = rng.normal(size=(n, 9, v))
    atomic = maps.transpose(1, 0, 2).reshape(9 * n, v)
    subject_gram = (atomic @ atomic.T).reshape(9, n, 9, n)
    weights = rng.exponential(size=(reps, 3, n))
    weights /= weights.sum(axis=2, keepdims=True)
    got_grams = production.group_grams_from_subject_gram(subject_gram, weights)
    direct_maps = np.einsum("rqn,nbv->rbqv", weights, maps,
                            optimize=True).reshape(reps, 27, v)
    expected_grams = np.einsum("riv,rjv->rij", direct_maps, direct_maps,
                               optimize=True)
    np.testing.assert_allclose(got_grams, expected_grams, rtol=3e-13,
                               atol=2e-12)
    got_anova = production.bootstrap_anova(
        got_grams, production.projection_matrices()
    )
    for replicate in range(reps):
        direct = production.functional_anova(
            expected_grams[replicate], production.projection_matrices()
        )
        for term in direct["ss"]:
            np.testing.assert_allclose(
                got_anova["term_ss"][term][replicate], direct["ss"][term],
                rtol=3e-13, atol=2e-12,
            )
        for ratio in direct["ratios"]:
            np.testing.assert_allclose(
                got_anova["ratios"][ratio][replicate], direct["ratios"][ratio],
                rtol=3e-13, atol=2e-12,
            )


def test_group_ols(rng: np.random.Generator) -> None:
    n_ei, n_er, v = 9, 6, 31
    y = rng.normal(size=(n_ei + n_er, v))
    is_ei = np.r_[np.ones(n_ei, dtype=bool), np.zeros(n_er, dtype=bool)]
    eligible = np.ones(len(is_ei), dtype=bool)
    intercept, t, df = production.group_intercept_t(y, eligible, is_ei)
    x = np.column_stack([np.ones(len(is_ei)), np.where(is_ei, 0.5, -0.5)])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    covariance00 = np.linalg.inv(x.T @ x)[0, 0]
    direct_t = beta[0] / np.sqrt(
        np.square(residual).sum(axis=0) / (len(is_ei) - 2) * covariance00
    )
    np.testing.assert_allclose(intercept, beta[0], rtol=3e-14, atol=2e-14)
    np.testing.assert_allclose(t, direct_t, rtol=5e-14, atol=2e-14)
    assert df == len(is_ei) - 2


def test_matched_deletion_metrics(rng: np.random.Generator) -> None:
    n_ei, n_er, v = 10, 10, 43
    n = n_ei + n_er
    groups = np.r_[np.zeros(n_ei, dtype=int), np.ones(n_er, dtype=int)]
    q0 = np.ones(n, dtype=bool)
    q1 = q0.copy(); q1[[n_ei, n_ei + 1]] = False
    q2 = q1.copy(); q2[[n_ei + 2, n_ei + 3, n_ei + 4]] = False
    observed_weights = np.stack([
        equal_cohort_weights(groups, mask) for mask in (q0, q1, q2)
    ])
    maps = rng.normal(size=(n, 9, v))
    atomic = maps.transpose(1, 0, 2).reshape(9 * n, v)
    subject_gram = (atomic @ atomic.T).reshape(9, n, 9, n)
    subject_sums = maps.sum(axis=2)
    observed_maps = np.einsum("qn,nbv->bqv", observed_weights, maps,
                              optimize=True)
    targets = {
        "Q0_to_Q1": np.zeros((production.N_DELETION, n)),
        "Q1_to_Q2": np.zeros((production.N_DELETION, n)),
    }
    for replicate in range(production.N_DELETION):
        target1 = q0.copy()
        target1[rng.choice(np.flatnonzero(q0 & (groups == 1)), 2,
                           replace=False)] = False
        target2 = q1.copy()
        target2[rng.choice(np.flatnonzero(q1 & (groups == 1)), 3,
                           replace=False)] = False
        targets["Q0_to_Q1"][replicate] = equal_cohort_weights(groups, target1)
        targets["Q1_to_Q2"][replicate] = equal_cohort_weights(groups, target2)
    _, distributions = production.matched_calibration(
        "synthetic", subject_gram, subject_sums, observed_maps,
        observed_weights, targets, v,
    )
    reference_norm = np.dot(observed_maps[0, 0], observed_maps[0, 0])
    for transition, base_q in (("Q0_to_Q1", 0), ("Q1_to_Q2", 1)):
        for b, (c, s) in enumerate(production.BASE_CELLS):
            random_maps = targets[transition] @ maps[:, b, :]
            base = observed_maps[b, base_q]
            direct_nrmsd = np.linalg.norm(random_maps - base, axis=1) / np.sqrt(
                reference_norm
            )
            centered_random = random_maps - random_maps.mean(axis=1,
                                                              keepdims=True)
            centered_base = base - base.mean()
            direct_one_minus_r = 1.0 - (
                centered_random @ centered_base
                / (np.linalg.norm(centered_random, axis=1)
                   * np.linalg.norm(centered_base))
            )
            label = f"{c}/{s}"
            np.testing.assert_allclose(
                distributions[("synthetic", transition,
                               "normalized_rms_difference", label)],
                direct_nrmsd, rtol=3e-12, atol=2e-13,
            )
            np.testing.assert_allclose(
                distributions[("synthetic", transition,
                               "one_minus_pearson_r", label)],
                direct_one_minus_r, rtol=3e-11, atol=2e-13,
            )


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(987654321))
    test_axis_labels_and_projections(rng)
    test_group_gram_order_and_bootstrap(rng)
    test_group_ols(rng)
    test_matched_deletion_metrics(rng)
    print("PASS: production aggregate axis/order, ANOVA/bootstrap, OLS-t, and "
          "matched-deletion Gram metrics equal direct voxel calculations")


if __name__ == "__main__":
    main()
