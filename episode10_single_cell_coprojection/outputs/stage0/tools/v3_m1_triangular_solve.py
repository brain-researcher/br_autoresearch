"""Solve with an existing Cholesky factor using NumPy's own ILP64 OpenBLAS.

The frozen EP10 environment has NumPy but not SciPy. Loading another BLAS
would change its numerical backend. Use the C BLAS ABI of the same library
already inspected by validate_v3_m1_numerical_execution_v3 instead.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _triangular_blas():
    libraries = sorted((Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("libopenblas*.so"))
    if len(libraries) != 1:
        raise RuntimeError("EP10 triangular solve requires NumPy's single OpenBLAS library")
    library = ctypes.CDLL(str(libraries[0]))
    config = library.openblas_get_config64_
    config.restype = ctypes.c_char_p
    if "USE64BITINT" not in config().decode("ascii", "strict"):
        raise RuntimeError("EP10 triangular solve requires the existing ILP64 ABI")
    solve = library.cblas_dtrsv64_
    solve.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                     ctypes.c_int64, ctypes.POINTER(ctypes.c_double), ctypes.c_int64,
                     ctypes.POINTER(ctypes.c_double), ctypes.c_int64]
    solve.restype = None
    return library, solve


def solve_cholesky(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return L^-T L^-1 rhs without altering L or rhs or refactoring L.

Copy into explicit C-contiguous binary64 buffers: neither a transposed nor a
strided caller array is passed to C with an incorrect leading dimension.
Only the lower triangle of the Cholesky factor is referenced. The caller
retains its unchanged two refinement steps and backward-error/descent gates.
"""
    matrix = np.asarray(factor, dtype=np.float64)
    vector = np.asarray(rhs, dtype=np.float64)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0 or vector.shape != (matrix.shape[0],)):
        raise ValueError("Cholesky triangular solve requires square L and one matching vector")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
        raise np.linalg.LinAlgError("Cholesky triangular solve received nonfinite values")
    if np.any(np.diag(matrix) <= 0.0):
        raise np.linalg.LinAlgError("Cholesky triangular solve requires a positive diagonal")
    matrix = np.array(matrix, dtype=np.float64, order="C", copy=True)
    result = np.array(vector, dtype=np.float64, order="C", copy=True)
    _, solve = _triangular_blas()
    n = matrix.shape[0]
    pointer = ctypes.POINTER(ctypes.c_double)
    # CBLAS RowMajor=101, Lower=122, NoTrans=111, Trans=112, NonUnit=131.
    for trans in (111, 112):
        solve(101, 122, trans, 131, n, matrix.ctypes.data_as(pointer), n,
              result.ctypes.data_as(pointer), 1)
    return result
