"""Dormant, bounded Finding A correlations over a complete supplied layer.

This array-only helper performs no data access, fitting, admission or random
draw generation, and launches no process or job. It can recompute one supplied
image resample. The caller authenticates all 224 tau-test image identities,
the common true features, final predictions, and fixed decoder-training feature
moments. Predictions/truth must use the tagged saved float32 representation,
not the bridge's restored float64 intermediate. Supplied callbacks enforce the
caller's resource/cancellation boundary; no access authority is created here.

BdPy 0.25 profiles correlate raw feature values across images. Patterns first
normalize both arrays using the decoder's float32 training moments, then
correlate across ALL layer units. Float64 centered-moment block merging avoids
the unstable difference of large raw second moments. It preserves this
estimand, not bitwise equality with a full-array or different-order reduction.
Finite correlations are clipped to [-1, 1], as in numpy.corrcoef; no epsilon,
NaN-column removal, Fisher transform or outcome-selected intersection is used.

Largest-layer result storage is about 55 MiB: one float64 profile vector, its
defined mask, and two float32 training-moment vectors. Working storage adds a
coverage mask and bounded 224-by-at-most-256 blocks, never a full feature bank.
Nineteen-layer summaries below are separate from paired contrasts: subtracting
them cannot replace each layer's exact paired defined-support check.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from decoder_interface import assembly
from schedule import PARTICIPANTS


TEST_IMAGES = 224
LAYERS = tuple(assembly.coverage.prefix.LAYERS)
_COMPARE_BLOCK = 262144


def _identity_image_indices():
    indices = np.arange(TEST_IMAGES, dtype=np.int64)
    indices.setflags(write=False)
    return indices


@dataclass(frozen=True)
class MetricBlock:
    decoder_participant: str
    image_ids: tuple[str, ...]
    full_layer_C_index: np.ndarray
    predicted: np.ndarray
    truth: np.ndarray
    training_mean: np.ndarray
    training_std: np.ndarray


@dataclass(frozen=True)
class CorrelationSummary:
    # Pattern positions identify draw occurrences through image_indices;
    # profile positions are full-layer C IDs.
    correlations: np.ndarray
    defined_mask: np.ndarray
    mean: float | None
    defined_count: int
    undefined_count: int

    @property
    def available(self):
        return self.mean is not None


@dataclass(frozen=True)
class LayerScores:
    method: str
    source: str
    target: str | None
    decoder_participant: str
    layer: str
    image_ids: tuple[str, ...]
    pattern: CorrelationSummary
    profile: CorrelationSummary
    decoder_training_image_ids: tuple[str, ...]
    training_mean: np.ndarray
    training_std: np.ndarray
    image_indices: np.ndarray = field(default_factory=_identity_image_indices)


@dataclass(frozen=True)
class MetricContrast:
    available: bool
    difference: float | None
    reason: str


@dataclass(frozen=True)
class LayerContrast:
    layer: str
    source: str
    target: str | None
    first_method: str
    second_method: str
    pattern: MetricContrast
    profile: MetricContrast


@dataclass(frozen=True)
class MetricSummary:
    available: bool
    mean: float | None
    reason: str


@dataclass(frozen=True)
class CrossLayerSummary:
    pattern: MetricSummary
    profile: MetricSummary


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _readonly(value):
    value.setflags(write=False)
    return value


def _ids(values, count):
    _require(isinstance(values, tuple) and len(values) == count
             and all(isinstance(value, str) and bool(value) for value in values)
             and len(set(values)) == count, "complete_unique_image_ID_tuple_required")


def _validate_image_indices(indices):
    _require(isinstance(indices, np.ndarray) and indices.shape == (TEST_IMAGES,)
             and indices.dtype.kind in "iu" and not indices.flags.writeable
             and bool((indices >= 0).all()) and bool((indices < TEST_IMAGES).all()),
             "readonly_224_integer_image_draw_positions_in_range_required")


def _owner(method, source, target):
    _require(source in PARTICIPANTS, "fixed_source_participant_required")
    if method == "within_source":
        _require(target is None, "within_source_is_reusable_without_target_binding")
        return source
    _require(method in ("content_loss", "brain_loss")
             and target in PARTICIPANTS and target != source,
             "fixed_nonself_conversion_method_pair_required")
    return target


def _unit_count(layer):
    _require(layer in LAYERS, "registered_VGG19_layer_required")
    return math.prod(assembly.coverage.prefix.SHAPES[LAYERS.index(layer)])


def _float32(value, shape, message):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype("float32")
             and value.shape == shape, message)


def _block(block, owner, image_ids, units, seen):
    _require(isinstance(block, MetricBlock) and block.decoder_participant == owner
             and block.image_ids == image_ids, "exact_decoder_owner_and_paired_image_order_required")
    ids = block.full_layer_C_index
    _require(isinstance(ids, np.ndarray) and ids.ndim == 1 and ids.dtype.kind in "iu"
             and 1 <= len(ids) <= assembly.MAX_BLOCK_UNITS
             and bool((ids >= 0).all()) and bool((ids < units).all())
             and len(set(ids.tolist())) == len(ids) and not bool(seen[ids].any()),
             "bounded_unique_unseen_full_layer_C_unit_IDs_required")
    for name in ("predicted", "truth"):
        _float32(getattr(block, name), (TEST_IMAGES, len(ids)),
                 "complete_saved_float32_prediction_and_truth_rows_required")
    for name in ("training_mean", "training_std"):
        value = getattr(block, name)
        _float32(value, (len(ids),), "frozen_float32_decoder_training_feature_moments_required")
        _require(bool(np.isfinite(value).all()), "finite_decoder_training_feature_moments_required")
    _require(bool((block.training_std > 0).all()), "positive_frozen_feature_scales_required")
    return ids


def _centered(x, y, axis):
    """Float64 centered sums; a nonfinite required value poisons its own row/unit."""
    xmean, ymean = np.mean(x, axis=axis), np.mean(y, axis=axis)
    xc = x - np.expand_dims(xmean, axis=axis)
    yc = y - np.expand_dims(ymean, axis=axis)
    return (xmean, ymean, np.sum(xc * xc, axis=axis),
            np.sum(yc * yc, axis=axis), np.sum(xc * yc, axis=axis))


def _correlations(xx, yy, xy):
    # Successive division follows corrcoef's variance normalization and avoids
    # an unnecessary product of second moments. Exactly zero stays undefined.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = xy / np.sqrt(xx) / np.sqrt(yy)
    valid = (xx > 0) & (yy > 0) & np.isfinite(result)
    result[~valid] = np.nan
    result[valid] = np.clip(result[valid], -1., 1.)
    return result


def _summary(correlations, check):
    check()
    defined = np.isfinite(correlations)
    count = int(np.count_nonzero(defined))
    # Sum in bounded slices; no whole-layer filtered copy or block weighting.
    partial = []
    for first in range(0, len(correlations), _COMPARE_BLOCK):
        check()
        values = correlations[first:first + _COMPARE_BLOCK]
        mask = defined[first:first + _COMPARE_BLOCK]
        partial.append(float(np.sum(values[mask], dtype=np.float64)))
    mean = math.fsum(partial) / count if count else None
    return CorrelationSummary(_readonly(correlations), _readonly(defined),
                              mean, count, len(correlations) - count)


def score_layer(blocks, *, method, source, target, layer, expected_image_ids,
                decoder_training_image_ids, check, image_indices=None):
    """Consume one complete layer exactly once, allowing arbitrary C-block order.

    Both pattern and profile retain undefined identities. A nonfinite raw
    value invalidates its image's pattern and its unit's profile. Overflow in
    float32 pattern normalization alone invalidates that pattern; the raw
    profile can remain defined. No affected column is removed from any row.
    Frozen moment snapshots remain available to bind same-decoder contrasts.
    Optional readonly image_indices selects 224 original row positions with
    repetitions retained. Both endpoints are recomputed from those rows;
    profiles are never resampled from previously computed correlations.
    image_ids still authenticates the original unique identities, while each
    pattern position refers to an occurrence in the recorded draw vector.
    """
    check()
    owner = _owner(method, source, target)
    units = _unit_count(layer)
    _ids(expected_image_ids, TEST_IMAGES)
    _ids(decoder_training_image_ids, 4712)
    _require(not set(expected_image_ids).intersection(decoder_training_image_ids),
             "test_decoder_training_image_overlap_forbidden")
    if image_indices is None:
        draw = _identity_image_indices()
    else:
        _validate_image_indices(image_indices)
        # Snapshot caller storage so a mutable alias cannot change this draw.
        draw = _readonly(image_indices.astype(np.int64, copy=True))
    seen = np.zeros(units, dtype=bool)
    profile = np.empty(units, dtype=np.float64)
    training_mean = np.empty(units, dtype=np.float32)
    training_std = np.empty(units, dtype=np.float32)
    count = 0
    mean_x = np.zeros(TEST_IMAGES, dtype=np.float64)
    mean_y = np.zeros(TEST_IMAGES, dtype=np.float64)
    xx = np.zeros(TEST_IMAGES, dtype=np.float64)
    yy = np.zeros(TEST_IMAGES, dtype=np.float64)
    xy = np.zeros(TEST_IMAGES, dtype=np.float64)
    for block in blocks:
        check()
        ids = _block(block, owner, expected_image_ids, units, seen)
        # Invalid arithmetic produces undefined correlations, without changing
        # the complete layer's feature or image denominator/support universe.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            predicted, truth = block.predicted[draw], block.truth[draw]
            raw_x, raw_y = predicted.astype(np.float64), truth.astype(np.float64)
            _, _, px, py, pxy = _centered(raw_x, raw_y, axis=0)
            profile[ids] = _correlations(px, py, pxy)
            check()
            # These two operations MUST occur before float64 conversion.
            x = ((predicted - block.training_mean) / block.training_std).astype(np.float64)
            y = ((truth - block.training_mean) / block.training_std).astype(np.float64)
            bx, by, bxx, byy, bxy = _centered(x, y, axis=1)
            total = count + len(ids)
            dx, dy = bx - mean_x, by - mean_y
            factor = count * len(ids) / total
            xx += bxx + dx * dx * factor
            yy += byy + dy * dy * factor
            xy += bxy + dx * dy * factor
            mean_x += dx * (len(ids) / total)
            mean_y += dy * (len(ids) / total)
        count = total
        training_mean[ids], training_std[ids] = block.training_mean, block.training_std
        seen[ids] = True
        check()
    check()
    _require(count == units and bool(seen.all()), "complete_all_channel_layer_C_unit_coverage_required")
    pattern = _summary(_correlations(xx, yy, xy), check)
    profile = _summary(profile, check)
    return LayerScores(method, source, target, owner, layer, expected_image_ids,
        pattern, profile, decoder_training_image_ids,
        _readonly(training_mean), _readonly(training_std), draw)


def _equal_arrays(first, second, check):
    if (not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray)
            or first.shape != second.shape or first.dtype != second.dtype or first.ndim != 1):
        return False
    for start in range(0, len(first), _COMPARE_BLOCK):
        check()
        if not np.array_equal(first[start:start + _COMPARE_BLOCK], second[start:start + _COMPARE_BLOCK]):
            return False
    return True


def _score_metadata(score):
    """Check result identity, not re-read or rescore previously consumed units."""
    _require(isinstance(score, LayerScores), "completed_layer_scores_required")
    _require(score.decoder_participant == _owner(score.method, score.source, score.target)
             and score.layer in LAYERS, "layer_score_method_decoder_binding_required")
    _ids(score.image_ids, TEST_IMAGES)
    _ids(score.decoder_training_image_ids, 4712)
    _validate_image_indices(score.image_indices)
    for summary in (score.pattern, score.profile):
        _require(isinstance(summary, CorrelationSummary)
                 and summary.defined_count >= 0 and summary.undefined_count >= 0
                 and ((summary.mean is None and summary.defined_count == 0)
                      or (summary.mean is not None and summary.defined_count > 0
                          and math.isfinite(summary.mean))), "defined_correlation_summary_required")


def _contrast(first, second, check):
    check()
    if not _equal_arrays(first.defined_mask, second.defined_mask, check):
        return MetricContrast(False, None, "different_defined_supports_no_intersection_permitted")
    if not first.available or not second.available:
        return MetricContrast(False, None, "all_required_correlations_undefined")
    return MetricContrast(True, first.mean - second.mean, "identical_defined_supports")


def paired_contrast(first, second, *, check):
    """Return first-minus-second only on identical complete defined supports.

    Within-source has no target binding and may serve each outgoing pair.
    Converted methods must share their exact source/target. When decoder owner
    matches, its training image identities and complete frozen moments must
    also match. Different owners legitimately use different training moments.
    """
    check()
    _score_metadata(first)
    _score_metadata(second)
    _require(first.layer == second.layer and first.source == second.source
             and first.image_ids == second.image_ids
             and _equal_arrays(first.image_indices, second.image_indices, check)
             and (first.target is None or second.target is None or first.target == second.target),
             "identical_layer_source_dyad_test_image_IDs_and_draw_positions_required")
    if first.decoder_participant == second.decoder_participant:
        _require(first.decoder_training_image_ids == second.decoder_training_image_ids
                 and _equal_arrays(first.training_mean, second.training_mean, check)
                 and _equal_arrays(first.training_std, second.training_std, check),
                 "same_decoder_requires_identical_training_images_and_feature_moments")
    return LayerContrast(first.layer, first.source,
        first.target if first.target is not None else second.target, first.method, second.method,
        _contrast(first.pattern, second.pattern, check), _contrast(first.profile, second.profile, check))


def summarize_layers(scores, *, check):
    """Equal-weight nineteen-layer method means, never a paired contrast.

    Consume previously computed LayerScores; the complete-coverage proof and
    per-unit vectors belong to score_layer, and are not recomputed here. Each
    metric is unavailable if any required layer's metric is unavailable.
    """
    check()
    collected = {}
    identity = None
    for score in scores:
        check()
        _score_metadata(score)
        current = (score.method, score.source, score.target, score.decoder_participant,
                   score.image_ids, score.decoder_training_image_ids,
                   tuple(score.image_indices.tolist()))
        _require(identity is None or identity == current,
                 "common_method_dyad_images_draw_positions_and_decoder_training_required")
        identity = current
        _require(score.layer not in collected, "duplicate_layer_summary_forbidden")
        collected[score.layer] = score
    if set(collected) != set(LAYERS):
        missing = MetricSummary(False, None, "all_nineteen_required_layers_must_be_present")
        return CrossLayerSummary(missing, missing)
    results = []
    for metric in ("pattern", "profile"):
        check()
        values = [getattr(collected[layer], metric) for layer in LAYERS]
        if any(not value.available for value in values):
            results.append(MetricSummary(False, None, "required_layer_metric_unavailable"))
        else:
            results.append(MetricSummary(True, math.fsum(value.mean for value in values) / len(LAYERS),
                                         "equal_weight_all_nineteen_layers"))
    return CrossLayerSummary(*results)
