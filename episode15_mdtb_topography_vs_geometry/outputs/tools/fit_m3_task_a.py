"""Array-only Task-A fit for the registered, externally fixed M3 active basis.

No reader, source-map construction, basis selection, Task-B input or scoring.
The caller must enforce roles and freeze the common fitting recipe, weights,
basis/capacity/locality and tolerances before supplying matched Task-A arrays.
"""
import numpy as np

from spatial_deformation_operators import _matrix, apply_m3


def _tolerance(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be an explicit real scalar in (0, 1)")
    if not np.isrealobj(value):
        raise ValueError(f"{name} must be an explicit real scalar in (0, 1)")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an explicit real scalar in (0, 1)") from error
    if not np.isfinite(result) or not 0 < result < 1:
        raise ValueError(f"{name} must be an explicit real scalar in (0, 1)")
    return result


def fit_m3_task_a(source_task_a, target_task_a, B, *, condition_weights,
                  fit_global_gain, rank_rtol, numerical_atol):
    """Fit row-wise ``g X [I+B(O-I)B.T]`` to matched, centered Task-A rows.

    Both maps are centered across their common equal-weight voxel axis. The
    supplied strictly positive condition weights are normalized to sum to one.
    ``O`` is the unique orthogonal Procrustes solution, including reflections;
    no determinant correction, rank truncation or singular completion occurs.
    Rank tests use the explicit relative tolerance, separately for the weighted
    active source design and source/target cross-product. An unidentified fit
    refuses instead of selecting an unregistered basis or tie rule.

    If requested by the common all-class rule, one positive scalar gain is fit
    jointly; otherwise it is exactly one. The caller applies the same returned
    O/gain to every spatially centered Task-B source row. Locality, provenance,
    frozen sign/order conventions and all empirical certificates remain the
    caller's responsibility. Returned diagnostics do not certify these facts.
    """
    rank_rtol = _tolerance(rank_rtol, "rank_rtol")
    numerical_atol = _tolerance(numerical_atol, "numerical_atol")
    if type(fit_global_gain) is not bool:
        raise ValueError("fit_global_gain must explicitly select the common boolean rule")
    source = _matrix(source_task_a, "source_task_a")
    target = _matrix(target_task_a, "target_task_a")
    basis = _matrix(B, "B")
    if source.shape != target.shape or source.shape[0] == 0 or source.shape[1] < 2:
        raise ValueError("Task-A maps must share nonempty condition and voxel axes")
    conditions, voxels = source.shape
    if basis.shape[0] != voxels or not 0 < basis.shape[1] < voxels:
        raise ValueError("B must have a nonempty centered active subspace on the common voxel axis")
    dimension = basis.shape[1]
    if conditions < dimension:
        raise ValueError("Task-A active source design is not full rank")
    weights = np.asarray(condition_weights)
    if weights.shape != (conditions,) or weights.dtype.kind not in "fiu":
        raise ValueError("condition_weights must be a real vector on the Task-A condition axis")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("condition_weights must be finite and strictly positive")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        identity = np.eye(dimension)
        basis_orthogonality = float(np.max(np.abs(basis.T@basis-identity)))
        basis_centering = float(np.max(np.abs(basis.sum(axis=0))))
        if max(basis_orthogonality, basis_centering) > numerical_atol:
            raise ValueError("B must already be centered and orthonormal at the supplied tolerance")
        source = source-source.mean(axis=1, keepdims=True)
        target = target-target.mean(axis=1, keepdims=True)
        weights = weights/weights.max()
        weights = weights/weights.sum()
        if np.any(weights == 0):
            raise ValueError("condition weights underflowed; no condition may be silently removed")
        root_weights = np.sqrt(weights)[:, None]
        active_source = root_weights*(source@basis)
        active_target = root_weights*(target@basis)
        source_singular = np.linalg.svd(active_source, compute_uv=False)
        if source_singular[-1] <= rank_rtol*source_singular[0]:
            raise ValueError("Task-A active source design is not full rank at rank_rtol")
        cross_product = active_source.T@active_target
        left, cross_singular, right_t = np.linalg.svd(cross_product, full_matrices=False)
        if cross_singular[-1] <= rank_rtol*cross_singular[0]:
            raise ValueError("Task-A orthogonal solution is not unique at rank_rtol")
        orthogonal = left@right_t
        orthogonality = float(max(np.max(np.abs(orthogonal.T@orthogonal-identity)),
                                  np.max(np.abs(orthogonal@orthogonal.T-identity))))
        if orthogonality > numerical_atol:
            raise ArithmeticError("fitted O exceeds the supplied orthogonality tolerance")
        predicted = apply_m3(source, basis, orthogonal)
        source_energy = float(np.sum(weights[:, None]*predicted*predicted))
        if not np.isfinite(source_energy) or source_energy <= 0:
            raise ArithmeticError("nonpositive or nonfinite fitted source energy")
        gain = float(np.sum(weights[:, None]*predicted*target)/source_energy) if fit_global_gain else 1.0
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("no attained positive-global-gain optimum; zero/negative gain is prohibited")
        residual = gain*predicted-target
        loss = float(np.sum(weights[:, None]*residual*residual))
        if not np.isfinite(loss):
            raise ArithmeticError("nonfinite Task-A fitting loss")
    return {
        "O": orthogonal,
        "gain": gain,
        "weighted_task_a_squared_error": loss,
        "source_design_singular_values": source_singular,
        "cross_product_singular_values": cross_singular,
        "basis_orthonormality_max_abs": basis_orthogonality,
        "basis_centering_max_abs": basis_centering,
        "O_orthogonality_max_abs": orthogonality,
    }
