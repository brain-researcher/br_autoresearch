"""Fail-closed fitted-T/U null sampling primitives for EP12.

This module is intentionally isolated from the running observed-search DAG.
It does not read a snapshot, write an artifact, or submit work.  A future null
driver must explicitly supply a validated development-only design, one fitted
T or U model for every focal-provider-type/side pair, the exact raw-to-
collapsed mapping, and a frozen seed manifest.

The sampler conditions on every observed neuron's typed-partner total.  Raw
non-typed endpoint counts are copied exactly.  Sampled collapsed typed counts
are expanded to raw typed partners using proportions pooled across both sides
of the same focal provider type, with a development-wide both-side fallback
only when that type has no observed mass in the collapsed bin.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .scientific_models import FittedPredictiveModel


NULL_REPLICATE_COUNT = 99
SEED_PROTOCOL = "ep12.fitted_t_u_null_seed.v1"
SEED_MANIFEST_SCHEMA = "ep12.fitted_t_u_null_seed_manifest.v1"
SAMPLER_PROTOCOL = "ep12.fitted_t_u_null_sampler.v1"


class FittedNullError(RuntimeError):
    """The fitted-null contract is incomplete, inconsistent, or unsafe."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FittedNullError("null metadata is not canonical JSON") from exc


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(_canonical_json(list(parts))).digest()[:8], "big")


def _validated_seed(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise FittedNullError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result < 2**64:
        raise FittedNullError(f"{name} must be in [0, 2**64)")
    return result


def _validated_sha256(value: Any, name: str) -> str:
    result = str(value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise FittedNullError(f"{name} must be a lowercase SHA-256 hex digest")
    return result


def freeze_null_seed_manifest(
    *,
    master_seed: int,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the policy-required 99 deterministic replicate seeds.

    ``binding`` should identify the design, selected generator, and selection
    decision.  It is embedded in the manifest and authenticated by both a
    binding digest and a manifest digest.  There is deliberately no option to
    request a different replicate count.
    """

    master_seed = _validated_seed(master_seed, "master_seed")
    if not isinstance(binding, Mapping):
        raise FittedNullError("seed binding must be a mapping")
    normalized_binding = json.loads(_canonical_json(dict(binding)).decode("utf-8"))
    binding_sha256 = _digest_json(normalized_binding)
    replicates = [
        {
            "replicate_index": index,
            "replicate_id": f"null_{index + 1:03d}",
            "seed": _stable_seed(
                SEED_PROTOCOL,
                master_seed,
                binding_sha256,
                "replicate",
                index,
            ),
        }
        for index in range(NULL_REPLICATE_COUNT)
    ]
    seeds = [row["seed"] for row in replicates]
    if len(set(seeds)) != NULL_REPLICATE_COUNT:
        raise FittedNullError("deterministic replicate seed collision")
    core: dict[str, Any] = {
        "schema_version": SEED_MANIFEST_SCHEMA,
        "seed_protocol": SEED_PROTOCOL,
        "replicate_count": NULL_REPLICATE_COUNT,
        "master_seed": master_seed,
        "binding": normalized_binding,
        "binding_sha256": binding_sha256,
        "replicates": replicates,
    }
    return {**core, "manifest_sha256": _digest_json(core)}


def verify_null_seed_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
) -> tuple[int, ...]:
    """Verify a frozen seed manifest and return its ordered seed vector."""

    if not isinstance(manifest, Mapping):
        raise FittedNullError("seed manifest must be a mapping")
    required = {
        "schema_version",
        "seed_protocol",
        "replicate_count",
        "master_seed",
        "binding",
        "binding_sha256",
        "replicates",
        "manifest_sha256",
    }
    if set(manifest) != required:
        raise FittedNullError("seed manifest fields differ from the frozen schema")
    if manifest["schema_version"] != SEED_MANIFEST_SCHEMA:
        raise FittedNullError("seed manifest schema mismatch")
    if manifest["seed_protocol"] != SEED_PROTOCOL:
        raise FittedNullError("seed derivation protocol mismatch")
    if manifest["replicate_count"] != NULL_REPLICATE_COUNT:
        raise FittedNullError("seed manifest must contain exactly 99 replicates")
    master_seed = _validated_seed(manifest["master_seed"], "master_seed")
    if not isinstance(manifest["binding"], Mapping):
        raise FittedNullError("seed manifest binding must be a mapping")
    binding = json.loads(_canonical_json(dict(manifest["binding"])).decode("utf-8"))
    binding_sha256 = _digest_json(binding)
    if manifest["binding_sha256"] != binding_sha256:
        raise FittedNullError("seed manifest binding digest mismatch")
    normalized_expected = json.loads(
        _canonical_json(dict(expected_binding)).decode("utf-8")
    )
    if binding != normalized_expected:
        raise FittedNullError("seed manifest is bound to a different null design or generator")
    core = {key: manifest[key] for key in required if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _digest_json(core):
        raise FittedNullError("seed manifest digest mismatch")
    rows = manifest["replicates"]
    if not isinstance(rows, list) or len(rows) != NULL_REPLICATE_COUNT:
        raise FittedNullError("seed manifest replicate table is malformed")
    seeds: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "replicate_index",
            "replicate_id",
            "seed",
        }:
            raise FittedNullError("seed manifest replicate row is malformed")
        if row["replicate_index"] != index or row["replicate_id"] != f"null_{index + 1:03d}":
            raise FittedNullError("seed manifest replicate ordering is not canonical")
        seed = _validated_seed(row["seed"], f"replicate seed {index}")
        expected_seed = _stable_seed(
            SEED_PROTOCOL,
            master_seed,
            binding_sha256,
            "replicate",
            index,
        )
        if seed != expected_seed:
            raise FittedNullError("seed manifest contains a non-derived replicate seed")
        seeds.append(seed)
    if len(set(seeds)) != NULL_REPLICATE_COUNT:
        raise FittedNullError("seed manifest contains duplicate replicate seeds")
    return tuple(seeds)


def _readonly_array(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _normalize_neuron_id(value: Any) -> int | str:
    if isinstance(value, (bool, np.bool_)):
        raise FittedNullError("neuron IDs cannot be booleans")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value:
        return value
    raise FittedNullError("neuron IDs must be nonempty strings or integers")


def _array_hash(digest: Any, name: str, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
    digest.update(_canonical_json(list(contiguous.shape)) + b"\0")
    digest.update(contiguous.tobytes(order="C"))


@dataclass(frozen=True)
class FittedNullDesign:
    """A validated, row-stable development design for fitted-null sampling."""

    neuron_ids: tuple[int | str, ...]
    focal_types: tuple[str, ...]
    sides: tuple[str, ...]
    covariates: np.ndarray
    raw_counts: np.ndarray
    raw_vocabulary: tuple[str, ...]
    collapsed_vocabulary: tuple[str, ...]
    raw_to_collapsed: np.ndarray
    collapsed_typed_mask: np.ndarray
    typed_raw_mask: np.ndarray
    design_sha256: str

    @classmethod
    def create(
        cls,
        *,
        neuron_ids: Sequence[int | str],
        focal_types: Sequence[str],
        sides: Sequence[str],
        covariates: np.ndarray,
        raw_counts: np.ndarray,
        raw_vocabulary: Sequence[str],
        collapsed_vocabulary: Sequence[str],
        raw_to_collapsed: Sequence[int],
        collapsed_typed_bins: Sequence[str],
    ) -> "FittedNullDesign":
        normalized_ids = tuple(_normalize_neuron_id(value) for value in neuron_ids)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise FittedNullError("neuron IDs must be unique")
        normalized_types = tuple(str(value) for value in focal_types)
        if any(not value for value in normalized_types):
            raise FittedNullError("focal provider types must be nonempty")
        normalized_sides = tuple(str(value) for value in sides)
        row_count = len(normalized_ids)
        if not row_count:
            raise FittedNullError("null design must contain at least one neuron")
        if len(normalized_types) != row_count or len(normalized_sides) != row_count:
            raise FittedNullError("identity, type, and side vectors must have equal length")
        if any(value not in {"L", "R"} for value in normalized_sides):
            raise FittedNullError("side values must be exactly L or R")
        for provider_type in sorted(set(normalized_types)):
            observed_sides = {
                side
                for focal_type, side in zip(normalized_types, normalized_sides)
                if focal_type == provider_type
            }
            if observed_sides != {"L", "R"}:
                raise FittedNullError(
                    f"focal type {provider_type!r} does not contain both body sides"
                )

        counts = np.asarray(raw_counts)
        if counts.ndim != 2 or counts.shape[0] != row_count:
            raise FittedNullError("raw counts must be a neuron-by-raw-partner matrix")
        if not np.issubdtype(counts.dtype, np.integer):
            if not np.all(np.isfinite(counts)) or not np.all(counts == np.round(counts)):
                raise FittedNullError("raw counts must be finite integers")
            counts = np.round(counts).astype(np.int64)
        else:
            counts = counts.astype(np.int64, copy=False)
        if np.any(counts < 0):
            raise FittedNullError("raw counts cannot be negative")
        if np.any(counts.sum(axis=1) <= 0):
            raise FittedNullError("every null-design neuron needs positive observed mass")

        raw_names = tuple(str(value) for value in raw_vocabulary)
        collapsed_names = tuple(str(value) for value in collapsed_vocabulary)
        if len(raw_names) != counts.shape[1] or not raw_names:
            raise FittedNullError("raw vocabulary does not match the raw count matrix")
        if len(collapsed_names) < 3:
            raise FittedNullError("collapsed vocabulary must contain at least three bins")
        if any(not value for value in raw_names + collapsed_names):
            raise FittedNullError("partner vocabulary labels must be nonempty")
        if len(set(raw_names)) != len(raw_names) or len(set(collapsed_names)) != len(
            collapsed_names
        ):
            raise FittedNullError("partner vocabulary labels must be unique")

        mapping = np.asarray(raw_to_collapsed)
        if mapping.ndim != 1 or len(mapping) != len(raw_names):
            raise FittedNullError("raw-to-collapsed mapping must assign every raw partner")
        if not np.issubdtype(mapping.dtype, np.integer):
            if not np.all(np.isfinite(mapping)) or not np.all(mapping == np.round(mapping)):
                raise FittedNullError("raw-to-collapsed indices must be integers")
            mapping = np.round(mapping).astype(np.int64)
        else:
            mapping = mapping.astype(np.int64, copy=False)
        if np.any(mapping < 0) or np.any(mapping >= len(collapsed_names)):
            raise FittedNullError("raw-to-collapsed mapping contains an invalid index")

        typed_bin_names = tuple(str(value) for value in collapsed_typed_bins)
        if not typed_bin_names or len(set(typed_bin_names)) != len(typed_bin_names):
            raise FittedNullError("collapsed typed bins must be a nonempty unique list")
        unknown_typed_bins = set(typed_bin_names) - set(collapsed_names)
        if unknown_typed_bins:
            raise FittedNullError("collapsed typed-bin list contains an unknown label")
        collapsed_typed = np.asarray(
            [value in set(typed_bin_names) for value in collapsed_names], dtype=bool
        )
        raw_typed = np.asarray([value.startswith("typed::") for value in raw_names], dtype=bool)
        if not raw_typed.any():
            raise FittedNullError("raw vocabulary contains no typed partner")
        for raw_index, collapsed_index in enumerate(mapping):
            if bool(raw_typed[raw_index]) != bool(collapsed_typed[collapsed_index]):
                raise FittedNullError(
                    "typed and non-typed endpoint mass cannot share a collapsed bin"
                )
        for collapsed_index in np.flatnonzero(collapsed_typed):
            if not np.any((mapping == collapsed_index) & raw_typed):
                raise FittedNullError("a declared collapsed typed bin has no raw typed partners")

        covariate_values = np.asarray(covariates, dtype=float)
        if covariate_values.ndim != 2 or covariate_values.shape[0] != row_count:
            raise FittedNullError("covariates must have one row per neuron")
        if not np.all(np.isfinite(covariate_values)):
            raise FittedNullError("covariates contain non-finite values")

        digest = hashlib.sha256()
        digest.update(SAMPLER_PROTOCOL.encode("ascii") + b"\0")
        digest.update(
            _canonical_json(
                {
                    "neuron_ids": normalized_ids,
                    "focal_types": normalized_types,
                    "sides": normalized_sides,
                    "raw_vocabulary": raw_names,
                    "collapsed_vocabulary": collapsed_names,
                }
            )
        )
        _array_hash(digest, "covariates", covariate_values.astype(np.float64))
        _array_hash(digest, "raw_counts", counts.astype(np.int64))
        _array_hash(digest, "raw_to_collapsed", mapping.astype(np.int64))
        _array_hash(digest, "collapsed_typed_mask", collapsed_typed.astype(np.uint8))
        return cls(
            neuron_ids=normalized_ids,
            focal_types=normalized_types,
            sides=normalized_sides,
            covariates=_readonly_array(covariate_values.astype(np.float64)),
            raw_counts=_readonly_array(counts.astype(np.int64)),
            raw_vocabulary=raw_names,
            collapsed_vocabulary=collapsed_names,
            raw_to_collapsed=_readonly_array(mapping.astype(np.int64)),
            collapsed_typed_mask=_readonly_array(collapsed_typed),
            typed_raw_mask=_readonly_array(raw_typed),
            design_sha256=digest.hexdigest(),
        )


@dataclass(frozen=True)
class ExpansionRule:
    collapsed_index: int
    raw_indices: tuple[int, ...]
    development_probabilities: tuple[float, ...]
    probabilities_by_type: Mapping[str, tuple[float, ...]]
    source_by_type: Mapping[str, str]


@dataclass(frozen=True)
class TypePooledExpansionPlan:
    """Both-side raw expansion probabilities frozen from the observed design."""

    design_sha256: str
    rules: Mapping[int, ExpansionRule]
    fallback_type_bin_pairs: tuple[tuple[str, str], ...]
    plan_sha256: str

    @classmethod
    def build(cls, design: FittedNullDesign) -> "TypePooledExpansionPlan":
        rules: dict[int, ExpansionRule] = {}
        fallback_pairs: list[tuple[str, str]] = []
        focal_types_array = np.asarray(design.focal_types, dtype=object)
        provider_types = sorted(set(design.focal_types))
        for collapsed_index in np.flatnonzero(design.collapsed_typed_mask):
            raw_indices_array = np.flatnonzero(
                (design.raw_to_collapsed == collapsed_index) & design.typed_raw_mask
            )
            raw_indices = tuple(int(value) for value in raw_indices_array)
            development_mass = design.raw_counts[:, raw_indices_array].sum(axis=0).astype(float)
            development_total = float(development_mass.sum())
            if not math.isfinite(development_total) or development_total <= 0:
                raise FittedNullError(
                    "a collapsed typed bin lacks development-wide raw expansion support"
                )
            development_probabilities = development_mass / development_total
            probabilities_by_type: dict[str, tuple[float, ...]] = {}
            source_by_type: dict[str, str] = {}
            for provider_type in provider_types:
                type_mask = focal_types_array == provider_type
                # FittedNullDesign has already proved this mask contains L and R.
                type_mass = design.raw_counts[type_mask][:, raw_indices_array].sum(axis=0).astype(
                    float
                )
                type_total = float(type_mass.sum())
                if math.isfinite(type_total) and type_total > 0:
                    probabilities = type_mass / type_total
                    source = "focal_type_pooled_both_sides"
                else:
                    probabilities = development_probabilities
                    source = "development_wide_pooled_both_sides_fallback"
                    fallback_pairs.append(
                        (provider_type, design.collapsed_vocabulary[int(collapsed_index)])
                    )
                if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
                    raise FittedNullError("raw expansion probabilities are invalid")
                if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-12):
                    raise FittedNullError("raw expansion probabilities do not sum to one")
                probabilities_by_type[provider_type] = tuple(
                    float(value) for value in probabilities
                )
                source_by_type[provider_type] = source
            rules[int(collapsed_index)] = ExpansionRule(
                collapsed_index=int(collapsed_index),
                raw_indices=raw_indices,
                development_probabilities=tuple(
                    float(value) for value in development_probabilities
                ),
                probabilities_by_type=MappingProxyType(probabilities_by_type),
                source_by_type=MappingProxyType(source_by_type),
            )
        frozen_rules = MappingProxyType(rules)
        frozen_fallbacks = tuple(sorted(fallback_pairs))
        payload = _expansion_plan_payload_parts(
            design.design_sha256,
            frozen_rules,
            frozen_fallbacks,
        )
        return cls(
            design_sha256=design.design_sha256,
            rules=frozen_rules,
            fallback_type_bin_pairs=frozen_fallbacks,
            plan_sha256=_digest_json(payload),
        )


def _expansion_plan_payload_parts(
    design_sha256: str,
    rules: Mapping[int, ExpansionRule],
    fallback_type_bin_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for key in sorted(rules):
        rule = rules[key]
        rows.append(
            {
                "collapsed_index": int(rule.collapsed_index),
                "raw_indices": [int(value) for value in rule.raw_indices],
                "development_probabilities": [
                    float(value) for value in rule.development_probabilities
                ],
                "probabilities_by_type": {
                    str(provider_type): [float(value) for value in probabilities]
                    for provider_type, probabilities in sorted(
                        rule.probabilities_by_type.items()
                    )
                },
                "source_by_type": {
                    str(provider_type): str(source)
                    for provider_type, source in sorted(rule.source_by_type.items())
                },
            }
        )
    return {
        "schema_version": "ep12.type_pooled_expansion_plan.v1",
        "design_sha256": str(design_sha256),
        "rules": rows,
        "fallback_type_bin_pairs": [
            [str(provider_type), str(collapsed_bin)]
            for provider_type, collapsed_bin in fallback_type_bin_pairs
        ],
    }


def _verify_expansion_plan(
    plan: TypePooledExpansionPlan,
    design: FittedNullDesign,
) -> str:
    if not isinstance(plan, TypePooledExpansionPlan):
        raise FittedNullError("raw expansion plan has the wrong type")
    if plan.design_sha256 != design.design_sha256:
        raise FittedNullError("raw expansion plan belongs to a different null design")
    recorded = _validated_sha256(plan.plan_sha256, "expansion plan SHA-256")
    payload = _expansion_plan_payload_parts(
        plan.design_sha256,
        plan.rules,
        plan.fallback_type_bin_pairs,
    )
    if _digest_json(payload) != recorded:
        raise FittedNullError("raw expansion plan digest mismatch")
    expected_plan = TypePooledExpansionPlan.build(design)
    expected_payload = _expansion_plan_payload_parts(
        expected_plan.design_sha256,
        expected_plan.rules,
        expected_plan.fallback_type_bin_pairs,
    )
    if payload != expected_payload or recorded != expected_plan.plan_sha256:
        raise FittedNullError(
            "raw expansion plan differs from deterministic type-pooled expansion"
        )
    expected_indices = {
        int(value) for value in np.flatnonzero(design.collapsed_typed_mask)
    }
    if set(plan.rules) != expected_indices:
        raise FittedNullError("raw expansion plan does not cover every typed bin exactly")
    expected_types = set(design.focal_types)
    for collapsed_index, rule in plan.rules.items():
        if int(rule.collapsed_index) != int(collapsed_index):
            raise FittedNullError("raw expansion rule key/index mismatch")
        expected_raw = tuple(
            int(value)
            for value in np.flatnonzero(
                (design.raw_to_collapsed == collapsed_index) & design.typed_raw_mask
            )
        )
        if tuple(rule.raw_indices) != expected_raw:
            raise FittedNullError("raw expansion rule targets the wrong raw partners")
        if set(rule.probabilities_by_type) != expected_types:
            raise FittedNullError("raw expansion probabilities omit a focal type")
        if set(rule.source_by_type) != expected_types:
            raise FittedNullError("raw expansion provenance omits a focal type")
        for probabilities in (
            rule.development_probabilities,
            *rule.probabilities_by_type.values(),
        ):
            values = np.asarray(probabilities, dtype=float)
            if (
                values.shape != (len(expected_raw),)
                or not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.isclose(float(values.sum()), 1.0, atol=1e-12)
            ):
                raise FittedNullError("raw expansion probabilities are invalid")
    return recorded


def _validate_generator(model: FittedPredictiveModel, design: FittedNullDesign) -> None:
    if not isinstance(model, FittedPredictiveModel):
        raise FittedNullError("generator must be an EP12 FittedPredictiveModel")
    if model.model_id not in {"T", "U_linear", "U_curved"}:
        raise FittedNullError("fitted null generator must be T, U_linear, or U_curved")
    collapsed_count = len(design.collapsed_vocabulary)
    latent_dimension = collapsed_count - 1
    basis = np.asarray(model.basis, dtype=float)
    coefficients = np.asarray(model.nuisance_coefficients, dtype=float)
    if basis.shape != (collapsed_count, latent_dimension):
        raise FittedNullError("generator basis does not match the collapsed vocabulary")
    if coefficients.shape != (design.covariates.shape[1] + 1, latent_dimension):
        raise FittedNullError("generator nuisance schema does not match the design")
    if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(coefficients)):
        raise FittedNullError("generator basis or nuisance coefficients are non-finite")
    if not np.allclose(basis.T @ basis, np.eye(latent_dimension), atol=1e-10):
        raise FittedNullError("generator basis is not orthonormal")
    if not np.allclose(basis.sum(axis=0), np.zeros(latent_dimension), atol=1e-10):
        raise FittedNullError("generator basis is not in the simplex sum-zero subspace")
    if model.model_id == "T":
        return
    means = np.asarray(model.prior_means, dtype=float)
    covariances = np.asarray(model.prior_covariances, dtype=float)
    weights = np.asarray(model.prior_weights, dtype=float)
    if means.ndim != 2 or means.shape[1] != latent_dimension or means.shape[0] < 1:
        raise FittedNullError("U generator prior means are malformed")
    if covariances.shape != (means.shape[0], latent_dimension, latent_dimension):
        raise FittedNullError("U generator prior covariances are malformed")
    if weights.shape != (means.shape[0],):
        raise FittedNullError("U generator prior weights are malformed")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
        raise FittedNullError("U generator prior contains non-finite values")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise FittedNullError("U generator prior weights must be finite and positive")
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-10):
        raise FittedNullError("U generator prior weights do not sum to one")
    for covariance in covariances:
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise FittedNullError("U generator covariance is not symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
            raise FittedNullError("U generator covariance is not positive semidefinite")


def fitted_generator_sha256(
    model: FittedPredictiveModel,
    design: FittedNullDesign,
) -> str:
    """Hash exactly the fitted state used by this sampler."""

    _validate_generator(model, design)
    digest = hashlib.sha256()
    digest.update(SAMPLER_PROTOCOL.encode("ascii") + b"\0")
    digest.update(model.model_id.encode("ascii") + b"\0")
    _array_hash(digest, "basis", np.asarray(model.basis, dtype=np.float64))
    _array_hash(
        digest,
        "nuisance_coefficients",
        np.asarray(model.nuisance_coefficients, dtype=np.float64),
    )
    if model.model_id != "T":
        _array_hash(digest, "prior_means", np.asarray(model.prior_means, dtype=np.float64))
        _array_hash(
            digest,
            "prior_covariances",
            np.asarray(model.prior_covariances, dtype=np.float64),
        )
        _array_hash(
            digest,
            "prior_weights",
            np.asarray(model.prior_weights, dtype=np.float64),
        )
    return digest.hexdigest()


def _validated_generator_map(
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    design: FittedNullDesign,
) -> dict[tuple[str, str], FittedPredictiveModel]:
    """Validate exact (focal provider type, side) generator coverage."""

    if not isinstance(generators, Mapping):
        raise FittedNullError("generators must be keyed by (focal provider type, side)")
    normalized: dict[tuple[str, str], FittedPredictiveModel] = {}
    for key, model in generators.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or not isinstance(key[0], str)
            or not key[0]
            or key[1] not in {"L", "R"}
        ):
            raise FittedNullError(
                "every generator key must be a nonempty provider type and side L/R"
            )
        normalized_key = (key[0], key[1])
        if normalized_key in normalized:
            raise FittedNullError("generator map contains a duplicate normalized key")
        _validate_generator(model, design)
        normalized[normalized_key] = model
    expected = set(zip(design.focal_types, design.sides))
    observed = set(normalized)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise FittedNullError(
            "generator key coverage differs from the null design: "
            f"missing={missing!r}, extra={extra!r}"
        )
    return normalized


def fitted_generator_manifest(
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    design: FittedNullDesign,
) -> tuple[tuple[dict[str, str], ...], str]:
    """Hash every fitted type/side generator in canonical key order."""

    normalized = _validated_generator_map(generators, design)
    rows = tuple(
        {
            "focal_type": provider_type,
            "side": side,
            "model_id": normalized[(provider_type, side)].model_id,
            "generator_sha256": fitted_generator_sha256(
                normalized[(provider_type, side)], design
            ),
        }
        for provider_type, side in sorted(normalized)
    )
    return rows, _digest_json(rows)


def null_seed_binding(
    design: FittedNullDesign,
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    expansion_plan: TypePooledExpansionPlan,
    *,
    selection_sha256: str,
) -> dict[str, Any]:
    """Build the minimum binding required before freezing null seeds."""

    generator_rows, generator_manifest_sha256 = fitted_generator_manifest(
        generators, design
    )
    expansion_plan_sha256 = _verify_expansion_plan(expansion_plan, design)
    return {
        "sampler_protocol": SAMPLER_PROTOCOL,
        "design_sha256": _validated_sha256(design.design_sha256, "design_sha256"),
        "generator_unit": "focal_provider_type_by_side",
        "generator_count": len(generator_rows),
        "generators": list(generator_rows),
        "generator_manifest_sha256": generator_manifest_sha256,
        "expansion_plan_sha256": expansion_plan_sha256,
        "selection_sha256": _validated_sha256(selection_sha256, "selection_sha256"),
    }


def _covariance_root_strict(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -1e-10:
        raise FittedNullError("U generator covariance is not positive semidefinite")
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))


def _typed_probabilities(
    model: FittedPredictiveModel,
    design: FittedNullDesign,
    row_index: int,
    replicate_seed: int,
) -> np.ndarray:
    design_row = np.concatenate(([1.0], design.covariates[row_index]))
    base = design_row @ np.asarray(model.nuisance_coefficients, dtype=float)
    residual = np.zeros_like(base)
    identity = (
        design.neuron_ids[row_index],
        design.focal_types[row_index],
        design.sides[row_index],
    )
    if model.model_id != "T":
        latent_seed = _stable_seed(
            SAMPLER_PROTOCOL,
            replicate_seed,
            *identity,
            "latent_residual",
        )
        generator = np.random.Generator(np.random.PCG64(latent_seed))
        weights = np.asarray(model.prior_weights, dtype=float)
        component = int(generator.choice(len(weights), p=weights))
        mean = np.asarray(model.prior_means[component], dtype=float)
        covariance = np.asarray(model.prior_covariances[component], dtype=float)
        residual = mean + _covariance_root_strict(covariance) @ generator.standard_normal(
            len(mean)
        )
    logits = (base + residual) @ np.asarray(model.basis, dtype=float).T
    typed_logits = logits[design.collapsed_typed_mask]
    if not np.all(np.isfinite(typed_logits)):
        raise FittedNullError("fitted generator produced non-finite typed logits")
    typed_logits = typed_logits - float(np.max(typed_logits))
    probabilities = np.exp(typed_logits)
    probability_sum = float(probabilities.sum())
    if not math.isfinite(probability_sum) or probability_sum <= 0:
        raise FittedNullError("fitted generator produced invalid typed probabilities")
    probabilities /= probability_sum
    if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-12):
        raise FittedNullError("typed probabilities do not sum to one")
    return probabilities


def _collapse_raw_counts(design: FittedNullDesign, raw_counts: np.ndarray) -> np.ndarray:
    collapsed = np.zeros(
        (raw_counts.shape[0], len(design.collapsed_vocabulary)), dtype=np.int64
    )
    for raw_index, collapsed_index in enumerate(design.raw_to_collapsed):
        collapsed[:, int(collapsed_index)] += raw_counts[:, raw_index]
    return collapsed


@dataclass(frozen=True)
class FittedNullSample:
    replicate_index: int
    replicate_id: str
    replicate_seed: int
    design_sha256: str
    generator_manifest_sha256: str
    expansion_plan_sha256: str
    generator_models_by_type_side: tuple[tuple[str, str, str], ...]
    neuron_ids: tuple[int | str, ...]
    focal_types: tuple[str, ...]
    sides: tuple[str, ...]
    covariates: np.ndarray
    raw_counts: np.ndarray
    collapsed_counts: np.ndarray
    fallback_allocations_with_positive_mass: int


def sample_fitted_null(
    design: FittedNullDesign,
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    expansion_plan: TypePooledExpansionPlan,
    *,
    seed_manifest: Mapping[str, Any],
    replicate_index: int,
    selection_sha256: str,
) -> FittedNullSample:
    """Sample one fitted T/U null replicate with a generator per type and side."""

    normalized_generators = _validated_generator_map(generators, design)
    expansion_plan_sha256 = _verify_expansion_plan(expansion_plan, design)
    if isinstance(replicate_index, bool) or not isinstance(
        replicate_index, (int, np.integer)
    ):
        raise FittedNullError("replicate_index must be an integer")
    replicate_index = int(replicate_index)
    if not 0 <= replicate_index < NULL_REPLICATE_COUNT:
        raise FittedNullError("replicate_index is outside the frozen 99-replicate set")
    binding = null_seed_binding(
        design,
        normalized_generators,
        expansion_plan,
        selection_sha256=selection_sha256,
    )
    seeds = verify_null_seed_manifest(seed_manifest, expected_binding=binding)
    replicate_seed = seeds[replicate_index]

    original_collapsed = _collapse_raw_counts(design, design.raw_counts)
    sampled_raw = np.array(design.raw_counts, copy=True)
    sampled_raw[:, design.typed_raw_mask] = 0
    sampled_collapsed_typed = np.zeros_like(original_collapsed)
    typed_collapsed_indices = np.flatnonzero(design.collapsed_typed_mask)
    typed_totals = design.raw_counts[:, design.typed_raw_mask].sum(axis=1).astype(np.int64)
    fallback_positive = 0

    for row_index, typed_total in enumerate(typed_totals):
        generator_key = (
            design.focal_types[row_index],
            design.sides[row_index],
        )
        model = normalized_generators[generator_key]
        probabilities = _typed_probabilities(
            model,
            design,
            row_index,
            replicate_seed,
        )
        identity = (
            design.neuron_ids[row_index],
            design.focal_types[row_index],
            design.sides[row_index],
        )
        count_seed = _stable_seed(
            SAMPLER_PROTOCOL,
            replicate_seed,
            *identity,
            "collapsed_typed_multinomial",
        )
        count_generator = np.random.Generator(np.random.PCG64(count_seed))
        sampled_counts = count_generator.multinomial(int(typed_total), probabilities)
        sampled_collapsed_typed[row_index, typed_collapsed_indices] = sampled_counts
        for collapsed_index, collapsed_count in zip(
            typed_collapsed_indices, sampled_counts
        ):
            rule = expansion_plan.rules.get(int(collapsed_index))
            if rule is None:
                raise FittedNullError("raw expansion rule is missing for a typed bin")
            provider_type = design.focal_types[row_index]
            expansion_probabilities = np.asarray(
                rule.probabilities_by_type[provider_type], dtype=float
            )
            source = rule.source_by_type[provider_type]
            if int(collapsed_count) > 0 and source == (
                "development_wide_pooled_both_sides_fallback"
            ):
                fallback_positive += 1
            expansion_seed = _stable_seed(
                SAMPLER_PROTOCOL,
                replicate_seed,
                *identity,
                "raw_expansion",
                design.collapsed_vocabulary[int(collapsed_index)],
            )
            expansion_generator = np.random.Generator(np.random.PCG64(expansion_seed))
            allocation = expansion_generator.multinomial(
                int(collapsed_count), expansion_probabilities
            )
            sampled_raw[row_index, np.asarray(rule.raw_indices, dtype=np.int64)] = allocation

    endpoint_mask = ~design.typed_raw_mask
    if not np.array_equal(
        sampled_raw[:, endpoint_mask], design.raw_counts[:, endpoint_mask]
    ):
        raise FittedNullError("null sampler changed an exact endpoint count")
    if not np.array_equal(
        sampled_raw[:, design.typed_raw_mask].sum(axis=1), typed_totals
    ):
        raise FittedNullError("null sampler changed a neuron's typed-partner total")
    if not np.array_equal(sampled_raw.sum(axis=1), design.raw_counts.sum(axis=1)):
        raise FittedNullError("null sampler changed a neuron's total outgoing count")
    sampled_collapsed = _collapse_raw_counts(design, sampled_raw)
    if not np.array_equal(
        sampled_collapsed[:, design.collapsed_typed_mask],
        sampled_collapsed_typed[:, design.collapsed_typed_mask],
    ):
        raise FittedNullError("raw expansion did not reproduce sampled collapsed typed counts")
    if not np.array_equal(
        sampled_collapsed[:, ~design.collapsed_typed_mask],
        original_collapsed[:, ~design.collapsed_typed_mask],
    ):
        raise FittedNullError("null sampler changed collapsed endpoint counts")

    return FittedNullSample(
        replicate_index=replicate_index,
        replicate_id=f"null_{replicate_index + 1:03d}",
        replicate_seed=replicate_seed,
        design_sha256=design.design_sha256,
        generator_manifest_sha256=binding["generator_manifest_sha256"],
        expansion_plan_sha256=expansion_plan_sha256,
        generator_models_by_type_side=tuple(
            (
                str(row["focal_type"]),
                str(row["side"]),
                str(row["model_id"]),
            )
            for row in binding["generators"]
        ),
        neuron_ids=design.neuron_ids,
        focal_types=design.focal_types,
        sides=design.sides,
        covariates=_readonly_array(design.covariates),
        raw_counts=_readonly_array(sampled_raw),
        collapsed_counts=_readonly_array(sampled_collapsed),
        fallback_allocations_with_positive_mass=fallback_positive,
    )
