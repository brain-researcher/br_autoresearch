"""Coordinatewise participant covariance between two matched replicates.

Participants must already be aligned across the two arrays. The explicit
participant axis is the only axis centered or reduced; all remaining feature
axes retain their supplied order and coordinates. No participant is dropped
or reordered, and negative covariance is preserved.

For n participants, U=sum_i((x_i-mean(x))*(y_i-mean(y)))/(n-1).
Equivalently, U=sum_{i<j}((x_i-x_j)*(y_i-y_j))/(n*(n-1)). This is half the
mean unordered-pair difference product, not that pair mean itself.

The function supplies the registered participant-variance scaling only. It
does not choose a feature contraction/normalization, geometry metric, ratio,
margin, uncertainty rule, independence assessment or scientific certificate.
There is no source reader, model fit, selection, or output writer.
"""
import numpy as np


def _finite_real(value, name):
    array = np.asarray(value)
    if array.ndim == 0 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric array with a participant axis")
    array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite in every participant and coordinate")
    return array


def cross_replicate_covariance(first, second, *, participant_axis=0):
    """Return float64 U with exactly the participant axis removed.

    Shape agreement is structural, not proof of participant identity or
    independent replicate errors. Those remain caller/data-role obligations.
    An unrepresentable floating-point result raises rather than being clipped
    or silently repaired.
    """
    first = _finite_real(first, "first")
    second = _finite_real(second, "second")
    if first.shape != second.shape:
        raise ValueError("replicates must have the same complete participant/feature shape")
    if (isinstance(participant_axis, (bool, np.bool_))
            or not isinstance(participant_axis, (int, np.integer))
            or not -first.ndim <= participant_axis < first.ndim):
        raise ValueError("participant_axis must identify an existing array axis")
    left = np.moveaxis(first, participant_axis, 0)
    right = np.moveaxis(second, participant_axis, 0)
    participants = left.shape[0]
    if participants < 2:
        raise ValueError("at least two matched participants are required")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        centered_left = left-left.mean(axis=0)
        centered_right = right-right.mean(axis=0)
        result = np.sum(centered_left*centered_right, axis=0)/(participants-1)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("nonfinite participant cross-replicate covariance")
    return np.asarray(result, dtype=np.float64)
