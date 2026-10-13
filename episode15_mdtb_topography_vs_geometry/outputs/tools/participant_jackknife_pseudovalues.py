"""Paired participant deletion for the registered coordinatewise covariance.

This is arithmetic on supplied, already matched replicate arrays. It makes no
model/nuisance refitting decision, contracts no feature axes, and supplies no
interval distribution, multiplicity rule, margin or scientific qualification.
Participant alignment and independent replicate errors remain caller duties.
There is no source reader, fitting, selection, or output writer.
"""
import numpy as np

from participant_cross_replicate_covariance import cross_replicate_covariance


def jackknife_pseudovalues(first, second, *, participant_axis=0):
    """Return ``statistic``, ``leave_one_out`` and ``pseudovalues`` as float64.

    ``statistic`` is the qualified covariance U with the participant axis
    removed. The other two arrays retain the input shape and participant order:
    entry i deletes that same participant from both replicates. Require n>=3
    so each deleted sample still supports the covariance denominator.

    With d_i=(x_i-mean(x))*(y_i-mean(y)), the centered cross-product sum after
    paired deletion is (n-1)*U - n*d_i/(n-1). Thus U_(-i) divides that sum by
    n-2, and P_i=n*U-(n-1)*U_(-i)=(n*d_i-U)/(n-2). This avoids n repeated
    centering/reduction passes and preserves negative estimates/pseudovalues.
    """
    statistic = cross_replicate_covariance(first, second, participant_axis=participant_axis)
    # The qualified helper above validates real finite shape and axis support.
    left = np.moveaxis(np.asarray(first, dtype=np.float64), participant_axis, 0)
    right = np.moveaxis(np.asarray(second, dtype=np.float64), participant_axis, 0)
    participants = left.shape[0]
    if participants < 3:
        raise ValueError("at least three matched participants are required for paired deletion")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        products = (left-left.mean(axis=0))*(right-right.mean(axis=0))
        deleted = ((participants-1)*statistic - (participants/(participants-1))*products)/(participants-2)
        pseudovalues = (participants*products-statistic)/(participants-2)
    if not np.isfinite(deleted).all() or not np.isfinite(pseudovalues).all():
        raise ArithmeticError("nonfinite paired participant deletion or pseudovalue")
    return {"statistic": statistic,
            "leave_one_out": np.moveaxis(deleted, 0, participant_axis),
            "pseudovalues": np.moveaxis(pseudovalues, 0, participant_axis)}
