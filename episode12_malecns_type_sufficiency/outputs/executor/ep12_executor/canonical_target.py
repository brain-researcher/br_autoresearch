"""Source-trained decoding from internal pools to the fixed count target.

The decoder has 0.5 prior count per canonical category, independent of the
model-fitting pseudocount. It is shared by T/U/M and refitted inside source CV.
Its conditional composition cannot recover latent differences within a pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.special import gammaln


DECODER_PROTOCOL = "ep12.source_count_canonical_decoder.v1"
DECODER_PSEUDOCOUNT = 0.5
CANONICAL_ENDPOINTS = (
    "unknown", "untyped", "fragment", "proofreading",
    "missing_annotation", "out_of_vocabulary",
)


def endpoint_bin(raw_key: str) -> str:
    if raw_key in CANONICAL_ENDPOINTS:
        return raw_key
    if raw_key.startswith("typed::"):
        return "other_typed"
    if not raw_key.startswith("untyped_status::"):
        return "out_of_vocabulary"
    fields = raw_key.split("::", 2)
    if len(fields) != 3:
        return "out_of_vocabulary"
    _, status, status_label = fields
    normalized = f"{status} {status_label}".lower()
    if "orphan" in normalized or "leaves" in normalized:
        return "fragment"
    if status in {"Assign", "Anchor"} or any(
        token in normalized
        for token in ("hard to trace", "partially traced", "anchor", "assign")
    ):
        return "proofreading"
    return "untyped" if status == "Traced" else "unknown"


def count_matrix(value: object) -> sparse.csr_matrix:
    counts = sparse.csr_matrix(value, dtype=float, copy=True)
    counts.sum_duplicates()
    if counts.ndim != 2 or counts.shape[1] < 1:
        raise ValueError("canonical counts must be a neuron-by-category matrix")
    if (not np.all(np.isfinite(counts.data)) or np.any(counts.data < 0)
            or not np.all(counts.data == np.round(counts.data))):
        raise ValueError("canonical counts must be finite nonnegative integers")
    return counts


def canonicalize_counts(
    counts: object,
    raw_vocabulary: Sequence[str],
    raw_to_pool: Sequence[int],
    pool_vocabulary: Sequence[str],
) -> tuple[sparse.csr_matrix, list[str], np.ndarray]:
    """Use annotation keys, including all six zero-observed semantic endpoints."""

    raw = count_matrix(counts)
    if raw.shape[1] != len(raw_vocabulary) or len(raw_to_pool) != len(raw_vocabulary):
        raise ValueError("raw vocabulary and pool mapping dimensions disagree")
    if len(set(raw_vocabulary)) != len(raw_vocabulary):
        raise ValueError("raw partner vocabulary contains duplicate identities")
    vocabulary = sorted(key for key in raw_vocabulary if key.startswith("typed::"))
    vocabulary += list(CANONICAL_ENDPOINTS)
    index = {key: position for position, key in enumerate(vocabulary)}
    pool_index = {key: position for position, key in enumerate(pool_vocabulary)}
    columns = []
    canonical_to_pool = np.full(len(vocabulary), -1, dtype=np.int64)
    for key, pool in zip(raw_vocabulary, raw_to_pool):
        canonical = index[key if key.startswith("typed::") else endpoint_bin(key)]
        if not 0 <= int(pool) < len(pool_vocabulary):
            raise ValueError("raw pool index is out of range")
        previous = canonical_to_pool[canonical]
        if previous != -1 and previous != int(pool):
            raise ValueError("one canonical category maps to different pools")
        canonical_to_pool[canonical] = int(pool)
        columns.append(canonical)
    for endpoint in CANONICAL_ENDPOINTS:
        if endpoint not in pool_index:
            raise ValueError("all six semantic endpoint pools must be retained")
        if canonical_to_pool[index[endpoint]] not in {-1, pool_index[endpoint]}:
            raise ValueError("semantic endpoint was mixed with a different pool")
        canonical_to_pool[index[endpoint]] = pool_index[endpoint]
    mapping = sparse.csr_matrix(
        (np.ones(len(columns)), (np.arange(len(columns)), columns)),
        shape=(len(raw_vocabulary), len(vocabulary)),
    )
    canonical = (raw @ mapping).tocsr()
    _pool_mapping(canonical_to_pool, len(vocabulary), len(pool_vocabulary))
    return canonical, vocabulary, canonical_to_pool


def _pool_mapping(mapping: object, categories: int, pools: int) -> np.ndarray:
    values = np.asarray(mapping)
    if (values.shape != (categories,) or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or np.any(values >= pools)):
        raise ValueError("canonical-to-pool mapping is invalid")
    if set(values.tolist()) != set(range(pools)):
        raise ValueError("structurally empty internal pool must be removed before fitting")
    return values.astype(np.int64, copy=True)


@dataclass(frozen=True)
class CanonicalDecoder:
    canonical_to_pool: np.ndarray
    log_conditional_probability: np.ndarray
    pool_count: int

    def __post_init__(self):
        mapping = _pool_mapping(
            self.canonical_to_pool, len(self.log_conditional_probability), self.pool_count
        )
        log_probabilities = np.asarray(self.log_conditional_probability, dtype=float).copy()
        if not np.all(np.isfinite(log_probabilities)):
            raise ValueError("canonical conditional probabilities must be finite")
        mass = np.bincount(mapping, weights=np.exp(log_probabilities), minlength=self.pool_count)
        if not np.allclose(mass, 1.0, atol=1e-12, rtol=0):
            raise ValueError("canonical conditional probabilities must normalize within each pool")
        mapping.setflags(write=False)
        log_probabilities.setflags(write=False)
        object.__setattr__(self, "canonical_to_pool", mapping)
        object.__setattr__(self, "log_conditional_probability", log_probabilities)

    @classmethod
    def fit(cls, training_counts: object, canonical_to_pool: object, pool_count: int):
        training = count_matrix(training_counts)
        mapping = _pool_mapping(canonical_to_pool, training.shape[1], pool_count)
        mass = np.asarray(training.sum(axis=0)).ravel() + DECODER_PSEUDOCOUNT
        pooled_mass = np.bincount(mapping, weights=mass, minlength=pool_count)
        log_probability = np.log(mass) - np.log(pooled_mass[mapping])
        mapping.setflags(write=False)
        log_probability.setflags(write=False)
        return cls(mapping, log_probability, pool_count)

    def pooled_counts(self, canonical_counts: object) -> np.ndarray:
        counts = count_matrix(canonical_counts)
        if counts.shape[1] != len(self.canonical_to_pool):
            raise ValueError("canonical target vocabulary differs from training")
        mapping = sparse.csr_matrix(
            (np.ones(counts.shape[1]), (np.arange(counts.shape[1]), self.canonical_to_pool)),
            shape=(counts.shape[1], self.pool_count),
        )
        return (counts @ mapping).toarray()

    def correction(self, canonical_counts: object, pooled_counts: object) -> np.ndarray:
        """Canonical-minus-pooled multinomial score per original count."""

        counts = count_matrix(canonical_counts)
        pooled = np.asarray(pooled_counts)
        if not np.array_equal(self.pooled_counts(counts), pooled):
            raise ValueError("canonical and internal counts do not conserve category mass")
        totals = np.asarray(counts.sum(axis=1)).ravel()
        if np.any(totals <= 0):
            raise ValueError("every scored neuron needs positive canonical mass")
        factorial = counts.copy()
        factorial.data = gammaln(factorial.data + 1.0)
        correction = (
            np.sum(gammaln(pooled + 1.0), axis=1)
            - np.asarray(factorial.sum(axis=1)).ravel()
            + np.asarray(counts @ self.log_conditional_probability).ravel()
        ) / totals
        if not np.all(np.isfinite(correction)):
            raise ValueError("canonical predictive correction is nonfinite")
        return correction

    def decode_probabilities(self, pooled_probabilities: object) -> np.ndarray:
        probabilities = np.asarray(pooled_probabilities, dtype=float)
        if (probabilities.shape[-1] != self.pool_count
                or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=-1), 1.0)):
            raise ValueError("internal probabilities must be normalized and nonnegative")
        return (probabilities[..., self.canonical_to_pool]
                * np.exp(self.log_conditional_probability))
