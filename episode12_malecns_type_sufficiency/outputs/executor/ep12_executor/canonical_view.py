"""One count-view transformation for observed and generated control branches."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse

from .canonical_target import CANONICAL_ENDPOINTS, endpoint_bin
from .development_trial import _collapse_vocabulary
from .falsifier_controls import pooled_multinomial_noise_sparse


def pool_counts(counts, mapping, width):
    mapping = np.asarray(mapping, dtype=np.int64)
    transform = sparse.csr_matrix(
        (np.ones(len(mapping), dtype=np.int64), (np.arange(len(mapping)), mapping)),
        shape=(len(mapping), width),
    )
    return (sparse.csr_matrix(counts, dtype=np.int64) @ transform).tocsr()


def expand_pooled_noise(pooled, original, mapping, *, seed):
    """Conditional allocation under the same globally pooled noise law.

    This preserves the original pooled draw exactly. It adds no within-pool
    structure: the conditional probabilities come from the global count
    totals that define this registered noise control, never a fitted T/U/M.
    """
    pooled = sparse.csr_matrix(pooled, dtype=np.int64)
    original = sparse.csr_matrix(original, dtype=np.int64)
    mapping = np.asarray(mapping, dtype=np.int64)
    mass = np.asarray(original.sum(axis=0)).ravel().astype(float)
    children = [np.flatnonzero(mapping == group) for group in range(pooled.shape[1])]
    probabilities = []
    for columns in children:
        values = mass[columns]
        probabilities.append(values / values.sum() if values.sum() else np.full(len(columns), 1 / len(columns)))
    rng = np.random.default_rng(int(seed))
    data, indices, indptr = [], [], [0]
    for row in range(pooled.shape[0]):
        for location in range(pooled.indptr[row], pooled.indptr[row + 1]):
            group, total = pooled.indices[location], int(pooled.data[location])
            allocation = rng.multinomial(total, probabilities[group])
            nonzero = np.flatnonzero(allocation)
            indices.extend(children[group][nonzero].tolist())
            data.extend(allocation[nonzero].tolist())
        indptr.append(len(data))
    result = sparse.csr_matrix((data, indices, indptr), shape=original.shape, dtype=np.int64)
    if (pool_counts(result, mapping, pooled.shape[1]) != pooled).nnz:
        raise ValueError("canonical noise expansion changed the frozen pooled draw")
    return result


def prepare_canonical_view(
    matrix: Any, focal_types: np.ndarray, raw_vocabulary: Sequence[str],
    configuration: Mapping[str, Any], hierarchy: Mapping[str, Mapping[str, str]] | None,
    *, matrix_transform: str = "identity", transform_seed: int = 0,
    selected_keys: Sequence[str] = (), endpoint: str | None = None,
    pool_name: str = "control_pool",
):
    matrix = sparse.csr_matrix(matrix, dtype=np.int64)
    raw_vocabulary = list(map(str, raw_vocabulary))
    selected = [index for index, key in enumerate(raw_vocabulary)
                if key in selected_keys or (endpoint is not None and endpoint_bin(key) == endpoint)]
    if selected_keys and not set(selected_keys) <= set(raw_vocabulary):
        raise ValueError("control partner keys are absent from the fixed annotation vocabulary")
    # Canonicalize before binary sensitivity, so multiple raw status aliases
    # cannot count as multiple semantic partners in observed data alone.
    semantic_names = sorted(key for key in raw_vocabulary if key.startswith("typed::"))
    semantic_names += list(CANONICAL_ENDPOINTS)
    semantic_index = {key: index for index, key in enumerate(semantic_names)}
    semantic_map = [semantic_index[key if key.startswith("typed::") else endpoint_bin(key)] for key in raw_vocabulary]
    canonical = pool_counts(matrix, semantic_map, len(semantic_names))
    transforms = []
    if matrix_transform == "binary":
        canonical = canonical.sign().astype(np.int64)
        transforms.append({"transform": "binary_canonical_partner_counts", "same_shape": True})
    elif matrix_transform not in {"identity", "pooled_multinomial_noise"}:
        raise ValueError("unsupported canonical matrix transform")
    selected_canonical = sorted(set(semantic_map[index] for index in selected))
    retained = canonical
    if selected_keys or endpoint is not None:
        if not selected_canonical or canonical[:, selected_canonical].sum() <= 0:
            raise ValueError("control pool has no observed mass")
        retained = canonical.tolil(copy=True)
        retained[:, selected_canonical] = 0
        retained = retained.tocsr()
    _, vocabulary, record = _collapse_vocabulary(
        retained, focal_types, semantic_names,
        strategy=str(configuration["partner_vocabulary"]),
        minimum_development_type_count=int(configuration["partner_minimum_development_type_count"]),
        rare_partner_mass_fraction=float(configuration["rare_partner_pooling_mass_fraction"]),
        hierarchy_depth=int(configuration["partner_hierarchy_depth"]),
        partner_hierarchy=None if hierarchy is None else dict(hierarchy), canonical_target=True,
    )
    mapping = np.asarray(record.pop("raw_to_pool"), dtype=np.int64)
    if selected_canonical:
        mapping[selected_canonical] = len(vocabulary)
        vocabulary = list(vocabulary) + [pool_name]
        occupied = np.unique(mapping)
        old_to_new = {int(old): new for new, old in enumerate(occupied)}
        mapping = np.asarray([old_to_new[int(old)] for old in mapping])
        vocabulary = [vocabulary[int(old)] for old in occupied]
        transforms.append({"transform": "explicit_retained_mass_pool",
                           "selected_canonical_keys": [semantic_names[i] for i in selected_canonical],
                           "row_totals_preserved": True, "total_mass_preserved": True})
    pooled = pool_counts(canonical, mapping, len(vocabulary))
    if matrix_transform == "pooled_multinomial_noise":
        noisy, receipt = pooled_multinomial_noise_sparse(pooled, seed=int(transform_seed))
        canonical = expand_pooled_noise(noisy, canonical, mapping, seed=int(transform_seed) ^ 0x43414E4F)
        pooled = noisy
        receipt["canonical_child_rule"] = "global_count_conditional_given_original_pooled_draw"
        transforms.append(receipt)
    record.update({"collapsed_partner_count": len(vocabulary),
                   "canonical_partner_count": len(semantic_names)})
    return pooled, canonical, mapping, list(vocabulary), record, transforms
