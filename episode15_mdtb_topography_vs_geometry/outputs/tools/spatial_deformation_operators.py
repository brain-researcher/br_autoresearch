"""Array-only registered M3/M4 spatial algebra, without fitted-model certification.

Rows of X are condition maps in one common, equally weighted voxel space.
B is the development-fixed, centered orthonormal voxel-by-active-axis basis;
O is orthogonal. C is the paired symmetric trace-free stretch/shear matrix.
These are caller preconditions, not conditions this module silently repairs.
Use raw_residuals to inspect their numerical residuals. No tolerance, rank,
locality, model-capacity or scientific acceptance threshold is supplied here.

Maps multiply operators on the RIGHT: Q=I+B(O-I)B.T and L=Q exp(B C B.T).
For the registered orthonormal B, L=I+B(O exp(C)-I)B.T. Orthogonal O may
include a reflection; this code does not add an unregistered determinant rule.
No centering, gain, fitting, input/output, or selection occurs in this module.
"""
import numpy as np
from scipy.linalg import expm


def _matrix(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric matrix")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _operators(B, O, C=None):
    basis, orthogonal = _matrix(B, "B"), _matrix(O, "O")
    voxels, dimension = basis.shape
    if voxels == 0 or dimension > voxels or orthogonal.shape != (dimension, dimension):
        raise ValueError("B/O dimensions do not define one voxel-space operator")
    stretch = None if C is None else _matrix(C, "C")
    if stretch is not None and stretch.shape != (dimension, dimension):
        raise ValueError("C must use the exact paired active basis")
    return basis, orthogonal, stretch


def _inputs(X, B, O, C=None):
    maps = _matrix(X, "X")
    basis, orthogonal, stretch = _operators(B, O, C)
    if maps.shape[0] == 0 or maps.shape[1] != basis.shape[0]:
        raise ValueError("X must contain condition rows on the same voxel axis as B")
    return maps, basis, orthogonal, stretch


def _finite(array):
    if not np.all(np.isfinite(array)):
        raise ArithmeticError("nonfinite spatial-operator arithmetic")
    return array


def _apply(maps, basis, active):
    with np.errstate(over="raise", invalid="raise"):
        return _finite(maps+(maps@basis)@(active-np.eye(basis.shape[1]))@basis.T)


def apply_m3(X, B, O):
    """Return XQ; B must be centered/orthonormal and O orthogonal to claim M3."""
    maps, basis, orthogonal, _ = _inputs(X, B, O)
    return _apply(maps, basis, orthogonal)


def apply_m4(X, B, O, C):
    """Return XL using the exact paired basis; zero C calls apply_m3 literally."""
    if C is None:
        raise ValueError("M4 requires an explicit paired C, including explicit zero stretch")
    maps, basis, orthogonal, stretch = _inputs(X, B, O, C)
    if not np.any(stretch):
        return apply_m3(maps, basis, orthogonal)
    with np.errstate(over="raise", invalid="raise"):
        active = _finite(orthogonal@expm(stretch))
    return _apply(maps, basis, active)


def inverse_m3(X, B, O):
    """Apply Q inverse (Q.T) under the registered centered-orthonormal B/O conditions."""
    maps, basis, orthogonal, _ = _inputs(X, B, O)
    return _apply(maps, basis, orthogonal.T)


def inverse_m4(X, B, O, C):
    """Apply L inverse = exp(-S) Q.T, not L.T; requires the registered B/O conditions."""
    if C is None:
        raise ValueError("M4 requires an explicit paired C, including explicit zero stretch")
    maps, basis, orthogonal, stretch = _inputs(X, B, O, C)
    if not np.any(stretch):
        return inverse_m3(maps, basis, orthogonal)
    with np.errstate(over="raise", invalid="raise"):
        active = _finite(expm(-stretch)@orthogonal.T)
    return _apply(maps, basis, active)


def _maximum(array):
    return float(np.max(np.abs(array))) if array.size else 0.0


def raw_residuals(B, O, C=None):
    """Raw algebra diagnostics only; deliberately no certified/passed Boolean.

    Q and the defining Q@exp(S) are assembled here so invalid-basis residuals
    stay visible. For C, the difference from the low-rank application formula
    is also reported: that shortcut requires the registered orthonormal basis.
    Rank identification, locality and model fitting are not evaluated.
    """
    basis, orthogonal, stretch = _operators(B, O, C)
    voxels, dimension = basis.shape
    identity, active_identity = np.eye(voxels), np.eye(dimension)
    ones = np.ones(voxels)
    with np.errstate(over="raise", invalid="raise"):
        Q = _finite(identity+basis@(orthogonal-active_identity)@basis.T)
        diagnostics = {
            "voxel_count": voxels, "basis_columns": dimension,
            "basis_orthonormality_max_abs": _maximum(basis.T@basis-active_identity),
            "basis_centering_max_abs": _maximum(basis.T@ones),
            "O_transpose_O_minus_I_max_abs": _maximum(orthogonal.T@orthogonal-active_identity),
            "O_O_transpose_minus_I_max_abs": _maximum(orthogonal@orthogonal.T-active_identity),
            "Q_transpose_Q_minus_I_max_abs": _maximum(Q.T@Q-identity),
            "Q_Q_transpose_minus_I_max_abs": _maximum(Q@Q.T-identity),
            "Q_constant_minus_constant_max_abs": _maximum(Q@ones-ones),
        }
        if stretch is not None:
            S = _finite(basis@stretch@basis.T)
            L = _finite(Q@expm(S))
            active_exponential = expm(stretch) if dimension else active_identity
            low_rank = _finite(identity+basis@(orthogonal@active_exponential-active_identity)@basis.T)
            diagnostics.update({
                "C_symmetry_max_abs": _maximum(stretch-stretch.T),
                "C_trace": float(np.trace(stretch)),
                "C_spectral_norm": float(np.linalg.norm(stretch, ord=2)) if dimension else 0.0,
                "S_constant_max_abs": _maximum(S@ones),
                "L_constant_minus_constant_max_abs": _maximum(L@ones-ones),
                "low_rank_vs_defining_L_max_abs": _maximum(low_rank-L),
            })
    if any(not np.isfinite(value) for value in diagnostics.values()):
        raise ArithmeticError("nonfinite spatial-operator diagnostic")
    return diagnostics
