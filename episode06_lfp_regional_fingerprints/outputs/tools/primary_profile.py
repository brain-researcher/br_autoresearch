"""EP06 primary residual-energy profiles, separate from component scoring.

For each frozen band block, S = 1 - sum((R - prediction)**2) / sum(R**2),
where R = (evaluation spikes - training direction/bin template) @ training
PCA loadings. Pool all whole-trial/native-bin/coordinate entries uniformly;
do not average coordinate scores, recenter evaluation residuals, clip negative
scores or apply component C/D/E projectors. Unequal direction counts therefore
retain their actual trial counts. This is fraction of training-template
residual energy recovered, not evaluation-centered explained variance.

Zero/nonfinite common residual energy makes every block unavailable. A
nonfinite prediction/error/ratio makes that block unavailable and the complete
profile unavailable; its identity is retained, never silently dropped. There
is no positive minimum-energy gate in this primary metric. Invalid metadata,
shapes or native observations are input errors, not missing measurements.

The paired result binds one session's identical evaluation trials, directions,
native bins, population dimension and ordered band blocks before subtracting
M1 minus PMd. Region-specific population-axis identities remain independent.
No classifier, profile normalization, pooling across sessions, resampling,
feature selection, file reader or scientific-opening authority is provided.
"""
from dataclasses import dataclass
import math

import numpy as np

from bandwise_prediction import (BandwisePredictions, FrozenPopulationContext, FeatureBlock,
                                _bind_fitted, transform_observations)
from component_scoring import BoundActivity, FrozenDesign


@dataclass(frozen=True)
class BlockScore:
    block: FeatureBlock
    squared_error: float | None
    score: float | None
    unavailable_reason: str | None

    @property
    def available(self):
        return self.score is not None


@dataclass(frozen=True)
class ProfileScore:
    region: str
    design: FrozenDesign                # evaluation rows, frozen population axes
    residual_energy: float | None       # unnormalized pooled sum over all entries
    blocks: tuple[BlockScore, ...]       # declared block order, including unavailable

    @property
    def available(self):
        return bool(self.blocks) and all(block.available for block in self.blocks)


@dataclass(frozen=True)
class SignedProfile:
    session_id: str
    blocks: tuple[FeatureBlock, ...]
    m1: ProfileScore
    pmd: ProfileScore
    m1_minus_pmd: tuple[float | None, ...]
    unavailable_reasons: tuple[str | None, ...]

    @property
    def available(self):
        return all(value is not None for value in self.m1_minus_pmd)


def _check(check):
    if not callable(check):
        raise TypeError("caller-owned cancellation/resource check required")
    check()


def _bind(fitted, observed, predictions):
    if not isinstance(fitted, FrozenPopulationContext) or not isinstance(observed, BoundActivity):
        raise TypeError("frozen bandwise fit and explicitly bound native observations required")
    if not isinstance(predictions, BandwisePredictions):
        raise TypeError("explicitly bound bandwise predictions required")
    if not isinstance(observed.design, FrozenDesign) or observed.design.role != "evaluation":
        raise PermissionError("primary profiles score evaluation whole trials only")
    design = _bind_fitted(fitted, observed.design, predictions.region)
    if (predictions.design != design or predictions.blocks != fitted.recipe.blocks
            or not isinstance(predictions.predictions, tuple)
            or len(predictions.predictions) != len(fitted.recipe.blocks)):
        raise ValueError("profile design and ordered blocks must match the frozen training fit")
    for prediction in predictions.predictions:
        if not isinstance(prediction, BoundActivity) or prediction.design != design:
            raise ValueError("every predicted block must retain exactly the bound evaluation design")
    return design


def _unavailable(region, design, blocks, energy, reason):
    return ProfileScore(region, design, energy,
                        tuple(BlockScore(block, None, None, reason) for block in blocks))


def score_band_profile(fitted, observed, predictions, *, check):
    """Score supplied frozen predictions against the training-template residual.

    ``observed`` carries native neuron axes and integer count values; its
    transformation reuses the fitted template/loadings without fitting. All
    prediction metadata binds before any observation or prediction value is
    read. Association with the genuine frozen fit remains the caller's duty.
    """
    _check(check)
    design = _bind(fitted, observed, predictions)
    # Check complete shapes without silently broadcasting a block. Numeric
    # nonfiniteness is reported below; malformed shapes/dtypes are input errors.
    arrays = []
    for prediction in predictions.predictions:
        _check(check)
        raw = np.asarray(prediction.values)
        if raw.dtype.kind not in "fiu" or raw.shape != design.shape:
            raise ValueError("every block needs its complete numeric trial/bin/population array")
        with np.errstate(over="ignore", invalid="ignore"):
            arrays.append(np.array(raw, dtype=np.float64, copy=True))
    try:
        residual = transform_observations(fitted, observed, region=predictions.region,
                                          check=check).values
    except FloatingPointError:
        return _unavailable(predictions.region, design, predictions.blocks, None,
                            "nonfinite_residual_energy")
    with np.errstate(over="ignore", invalid="ignore"):
        energy = float(np.sum(np.square(residual), dtype=np.float64))
    if not math.isfinite(energy) or energy <= 0:
        return _unavailable(predictions.region, design, predictions.blocks,
            energy if math.isfinite(energy) else None,
            "zero_residual_energy" if energy == 0 else "nonfinite_residual_energy")
    scores = []
    for block, prediction in zip(predictions.blocks, arrays):
        _check(check)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            error = float(np.sum(np.square(residual - prediction), dtype=np.float64))
            score = 1. - error / energy
        if not math.isfinite(error) or not math.isfinite(score):
            scores.append(BlockScore(block, error if math.isfinite(error) else None,
                                     None, "nonfinite_prediction_error_or_score"))
        else:
            scores.append(BlockScore(block, error, score, None))
    _check(check)
    return ProfileScore(predictions.region, design, energy, tuple(scores))


def paired_profile(m1, pmd, *, check):
    """Return ordered M1-minus-PMd scores without fitting or normalization."""
    _check(check)
    if (not isinstance(m1, ProfileScore) or not isinstance(pmd, ProfileScore)
            or m1.region != "M1" or pmd.region != "PMd"):
        raise ValueError("one explicitly ordered M1/PMd profile pair is required")
    left, right = m1.design, pmd.design
    if (not isinstance(left, FrozenDesign) or not isinstance(right, FrozenDesign)
            or left.role != "evaluation" or right.role != "evaluation"):
        raise PermissionError("paired primary profiles require evaluation designs")
    if (any(getattr(left, name) != getattr(right, name) for name in
            ("session_id", "directions", "trial_ids", "trial_directions", "time_ids"))
            or len(left.axis_ids) != len(right.axis_ids)):
        raise ValueError("paired profiles need exact session/trial/direction/time identities and equal rank")
    blocks = tuple(value.block for value in m1.blocks)
    if not blocks or blocks != tuple(value.block for value in pmd.blocks):
        raise ValueError("paired profiles must retain identical ordered feature-block identities")
    differences, reasons = [], []
    for a, b in zip(m1.blocks, pmd.blocks):
        _check(check)
        if a.score is None or b.score is None:
            differences.append(None)
            reasons.append("required_regional_block_unavailable")
            continue
        difference = a.score - b.score
        if not all(math.isfinite(value) for value in (a.score, b.score, difference)):
            differences.append(None)
            reasons.append("nonfinite_signed_profile_difference")
        else:
            differences.append(difference)
            reasons.append(None)
    _check(check)
    return SignedProfile(left.session_id, blocks, m1, pmd, tuple(differences), tuple(reasons))
