#!/usr/bin/env python3
"""Independent synthetic checks for the frozen NARPS analysis mathematics.

This is an audit-only test artifact.  It does not read neural outcomes and is
not imported by the production analysis.
"""

from __future__ import annotations

import itertools

import numpy as np


LEVELS = 3
U = np.column_stack(
    [
        np.ones(LEVELS) / np.sqrt(LEVELS),
        np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0),
    ]
)
J = np.ones((LEVELS, LEVELS)) / LEVELS
H = np.eye(LEVELS) - J


def projections() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    p0 = np.kron(np.kron(J, J), J)
    terms = {
        "C": np.kron(np.kron(H, J), J),
        "Q": np.kron(np.kron(J, H), J),
        "S": np.kron(np.kron(J, J), H),
        "CQ": np.kron(np.kron(H, H), J),
        "CS": np.kron(np.kron(H, J), H),
        "QS": np.kron(np.kron(J, H), H),
        "CQS": np.kron(np.kron(H, H), H),
    }
    return p0, terms


def direct_marginal_ss(y: np.ndarray) -> dict[str, float]:
    """Reference marginal-mean formulas for y[C,Q,S,V]."""
    grand = y.mean(axis=(0, 1, 2))
    ec = y.mean(axis=(1, 2)) - grand
    eq = y.mean(axis=(0, 2)) - grand
    es = y.mean(axis=(0, 1)) - grand
    ecq = (
        y.mean(axis=2)
        - y.mean(axis=(1, 2))[:, None, :]
        - y.mean(axis=(0, 2))[None, :, :]
        + grand
    )
    ecs = (
        y.mean(axis=1)
        - y.mean(axis=(1, 2))[:, None, :]
        - y.mean(axis=(0, 1))[None, :, :]
        + grand
    )
    eqs = (
        y.mean(axis=0)
        - y.mean(axis=(0, 2))[:, None, :]
        - y.mean(axis=(0, 1))[None, :, :]
        + grand
    )
    ecqs = (
        y
        - grand
        - ec[:, None, None, :]
        - eq[None, :, None, :]
        - es[None, None, :, :]
        - ecq[:, :, None, :]
        - ecs[:, None, :, :]
        - eqs[None, :, :, :]
    )
    return {
        "C": 9.0 * np.square(ec).sum(),
        "Q": 9.0 * np.square(eq).sum(),
        "S": 9.0 * np.square(es).sum(),
        "CQ": 3.0 * np.square(ecq).sum(),
        "CS": 3.0 * np.square(ecs).sum(),
        "QS": 3.0 * np.square(eqs).sum(),
        "CQS": np.square(ecqs).sum(),
    }


def equal_cohort_weights(
    raw: np.ndarray, groups: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    """Return 1/2-normalized EI plus 1/2-normalized ER coefficients."""
    result = np.zeros_like(raw, dtype=float)
    for group in (0, 1):
        use = eligible & (groups == group)
        denominator = raw[use].sum()
        if not use.any() or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Each eligible cohort needs positive total weight")
        result[use] = 0.5 * raw[use] / denominator
    return result


def aggregate_cs_grams(effects: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Build four exact C/S-subspace subject Gram matrices.

    effects has shape [subject,C,S,voxel].  Status 0 denotes the one-dimensional
    constant subspace and status 1 denotes the two-dimensional contrast subspace.
    """
    transformed = np.einsum(
        "ca,sb,ncsv->nabv", U, U, effects, optimize=True
    )
    indices = {0: (0,), 1: (1, 2)}
    result = {}
    for c_status, s_status in itertools.product((0, 1), repeat=2):
        selected = transformed[
            :, indices[c_status], :, :
        ][:, :, indices[s_status], :]
        result[(c_status, s_status)] = np.einsum(
            "nadv,madv->nm", selected, selected, optimize=True
        )
    return result


def qform(vector: np.ndarray, gram: np.ndarray) -> float:
    return float(vector @ gram @ vector)


def gram_anova_ss(effects: np.ndarray, q_weights: np.ndarray) -> dict[str, float]:
    """Exact ANOVA SS from subject maps and Q-level coefficient vectors."""
    grams = aggregate_cs_grams(effects)
    h = U.T @ q_weights
    q0 = lambda gram: qform(h[0], gram)
    q1 = lambda gram: sum(qform(h[index], gram) for index in (1, 2))
    return {
        "C": q0(grams[(1, 0)]),
        "Q": q1(grams[(0, 0)]),
        "S": q0(grams[(0, 1)]),
        "CQ": q1(grams[(1, 0)]),
        "CS": q0(grams[(1, 1)]),
        "QS": q1(grams[(0, 1)]),
        "CQS": q1(grams[(1, 1)]),
    }


def gram_pair_metrics(
    effects: np.ndarray,
    weight_a: np.ndarray,
    weight_b: np.ndarray,
    reference_effects: np.ndarray,
    reference_weights: np.ndarray,
) -> tuple[float, float]:
    """NRMSD and 1-r from raw and spatially centered subject Grams."""
    voxel_count = effects.shape[1]
    raw_gram = effects @ effects.T
    difference = weight_a - weight_b
    difference_ss = qform(difference, raw_gram)
    reference_gram = reference_effects @ reference_effects.T
    reference_ss = qform(reference_weights, reference_gram)
    if reference_ss <= 0:
        raise ValueError("Zero fixed-reference RMS denominator")
    nrmsd = np.sqrt(max(0.0, difference_ss) / reference_ss)

    spatial_sums = effects.sum(axis=1)
    centered_gram = raw_gram - np.outer(spatial_sums, spatial_sums) / voxel_count
    aa = qform(weight_a, centered_gram)
    bb = qform(weight_b, centered_gram)
    ab = float(weight_a @ centered_gram @ weight_b)
    if aa <= 0 or bb <= 0:
        raise ValueError("Pearson correlation has zero spatial variance")
    correlation = np.clip(ab / np.sqrt(aa * bb), -1.0, 1.0)
    return float(nrmsd), float(1.0 - correlation)


def test_projection_algebra(rng: np.random.Generator) -> None:
    assert np.allclose(U.T @ U, np.eye(3), atol=1e-14)
    assert np.allclose(U[:, [0]] @ U[:, [0]].T, J, atol=1e-14)
    assert np.allclose(U[:, 1:] @ U[:, 1:].T, H, atol=1e-14)
    p0, terms = projections()
    expected_ranks = {"C": 2, "Q": 2, "S": 2, "CQ": 4,
                      "CS": 4, "QS": 4, "CQS": 8}
    all_projectors = {"grand": p0, **terms}
    for name, projector in all_projectors.items():
        assert np.allclose(projector, projector.T, atol=1e-14), name
        assert np.allclose(projector @ projector, projector, atol=1e-14), name
    for left, right in itertools.combinations(all_projectors, 2):
        assert np.allclose(
            all_projectors[left] @ all_projectors[right], 0.0, atol=1e-14
        ), (left, right)
    assert np.allclose(p0 + sum(terms.values()), np.eye(27), atol=1e-14)
    assert {name: round(np.trace(p)) for name, p in terms.items()} == expected_ranks

    y = rng.normal(size=(3, 3, 3, 41))
    flat = y.reshape(27, 41)
    projected_ss = {
        name: float(np.square(projector @ flat).sum())
        for name, projector in terms.items()
    }
    marginal_ss = direct_marginal_ss(y)
    for name in terms:
        assert np.allclose(projected_ss[name], marginal_ss[name], rtol=2e-14), name
    total = np.square((np.eye(27) - p0) @ flat).sum()
    assert np.allclose(sum(projected_ss.values()), total, rtol=2e-14)


def test_shapley(rng: np.random.Generator) -> None:
    labels = ("C", "Q", "S")
    term_ss = {
        frozenset(("C",)): rng.uniform(),
        frozenset(("Q",)): rng.uniform(),
        frozenset(("S",)): rng.uniform(),
        frozenset(("C", "Q")): rng.uniform(),
        frozenset(("C", "S")): rng.uniform(),
        frozenset(("Q", "S")): rng.uniform(),
        frozenset(("C", "Q", "S")): rng.uniform(),
    }
    closed_form = {
        factor: sum(value / len(term) for term, value in term_ss.items()
                    if factor in term)
        for factor in labels
    }

    def game(coalition: set[str]) -> float:
        return sum(value for term, value in term_ss.items()
                   if term.issubset(coalition))

    enumerated = dict.fromkeys(labels, 0.0)
    permutations = tuple(itertools.permutations(labels))
    for order in permutations:
        coalition: set[str] = set()
        before = 0.0
        for factor in order:
            coalition.add(factor)
            after = game(coalition)
            enumerated[factor] += (after - before) / len(permutations)
            before = after
    for factor in labels:
        assert np.allclose(enumerated[factor], closed_form[factor], rtol=1e-14)
    assert np.allclose(sum(closed_form.values()), sum(term_ss.values()), rtol=1e-14)


def test_group_coefficients(rng: np.random.Generator) -> None:
    n_ei, n_er, voxel_count = 7, 4, 23
    maps = rng.normal(size=(n_ei + n_er, voxel_count))
    task_code = np.r_[np.full(n_ei, 0.5), np.full(n_er, -0.5)]
    design = np.column_stack([np.ones(n_ei + n_er), task_code])
    coefficients = np.linalg.lstsq(design, maps, rcond=None)[0]
    intercept_weights = np.r_[np.full(n_ei, 0.5 / n_ei),
                              np.full(n_er, 0.5 / n_er)]
    difference_weights = np.r_[np.full(n_ei, 1.0 / n_ei),
                               np.full(n_er, -1.0 / n_er)]
    assert np.allclose(coefficients[0], intercept_weights @ maps, rtol=1e-14)
    assert np.allclose(coefficients[1], difference_weights @ maps, rtol=1e-14)
    assert np.allclose(coefficients[0],
                       0.5 * (maps[:n_ei].mean(0) + maps[n_ei:].mean(0)))


def test_bootstrap_gram_equivalence(rng: np.random.Generator) -> None:
    subject_count, voxel_count = 12, 47
    groups = np.r_[np.zeros(6, dtype=int), np.ones(6, dtype=int)]
    eligible = np.ones((3, subject_count), dtype=bool)
    eligible[1, [7]] = False
    eligible[2, [7, 9]] = False
    raw = rng.exponential(size=subject_count)
    weights = np.stack(
        [equal_cohort_weights(raw, groups, eligible[q]) for q in range(3)]
    )
    effects = rng.normal(size=(subject_count, 3, 3, voxel_count))
    group_maps = np.einsum("qn,ncsv->cqsv", weights, effects, optimize=True)
    flat = group_maps.reshape(27, voxel_count)
    _, terms = projections()
    direct = {
        name: float(np.square(projector @ flat).sum())
        for name, projector in terms.items()
    }
    reduced = gram_anova_ss(effects, weights)
    for name in terms:
        assert np.allclose(direct[name], reduced[name], rtol=3e-13, atol=1e-11), (
            name, direct[name], reduced[name]
        )
    assert np.allclose(sum(direct.values()), sum(reduced.values()), rtol=2e-13)


def test_matched_deletion_gram_metrics(rng: np.random.Generator) -> None:
    subject_count, voxel_count = 15, 53
    groups = np.r_[np.zeros(7, dtype=int), np.ones(8, dtype=int)]
    source_eligible = np.ones(subject_count, dtype=bool)
    deleted = rng.choice(np.flatnonzero(groups == 1), size=2, replace=False)
    target_eligible = source_eligible.copy()
    target_eligible[deleted] = False
    ones = np.ones(subject_count)
    source_weight = equal_cohort_weights(ones, groups, source_eligible)
    target_weight = equal_cohort_weights(ones, groups, target_eligible)
    effects = rng.normal(size=(subject_count, voxel_count))
    reference_effects = rng.normal(size=(subject_count, voxel_count))
    reference_weight = source_weight.copy()

    map_a = source_weight @ effects
    map_b = target_weight @ effects
    reference = reference_weight @ reference_effects
    direct_nrmsd = np.sqrt(np.mean(np.square(map_a - map_b))) / np.sqrt(
        np.mean(np.square(reference))
    )
    direct_one_minus_r = 1.0 - np.corrcoef(map_a, map_b)[0, 1]
    gram_nrmsd, gram_one_minus_r = gram_pair_metrics(
        effects, source_weight, target_weight,
        reference_effects, reference_weight
    )
    assert np.allclose(gram_nrmsd, direct_nrmsd, rtol=2e-13, atol=1e-14)
    assert np.allclose(gram_one_minus_r, direct_one_minus_r,
                       rtol=2e-12, atol=1e-14)


def main() -> None:
    rng = np.random.Generator(np.random.PCG64(123456789))
    test_projection_algebra(rng)
    test_shapley(rng)
    test_group_coefficients(rng)
    for _ in range(20):
        test_bootstrap_gram_equivalence(rng)
        test_matched_deletion_gram_metrics(rng)
    print("PASS: projection, Shapley, group-coefficient, bootstrap-Gram, "
          "and deletion-metric invariants")


if __name__ == "__main__":
    main()
