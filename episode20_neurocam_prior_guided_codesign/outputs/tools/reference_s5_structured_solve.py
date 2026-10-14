"""Exact separable capacitance solve for the conditional full-state air circuit.

The rectangular pad block is PAD_C*I + PAD_NEIGHBOUR_C*(Lr ⊕ Lc),
with path-graph Laplacians. An orthonormal DCT-II diagonalizes each path.
The ideal driven pad is deleted by the identity
  C_free^-1 b = (M^-1 b0 - M^-1 e_d * (M^-1 b0)_d/(M^-1 e_d)_d)_free.
This is a Dirichlet constraint on the full inverse, not model reduction.
Source capacitance is another path block; remaining states are diagonal.
Only the linear algebra changes. All states, edges and time intervals remain.
"""
import numpy as np
import scipy.fft as fft
import scipy.sparse as sp

from reference_s5_family import (AirFamily, PAD_C, PAD_NEIGHBOUR_C,
    SOURCE_C, INPUT_C, BUFFER_C, FILTER_C, HOLD_CAPACITANCE_FARAD)
from reference_s5_cached_expm import CachedAirFamily


def _path_laplacian(n):
    if n == 1:
        return sp.csr_matrix((1, 1))
    diagonal = np.full(n, 2.)
    diagonal[[0, -1]] = 1.
    return sp.diags((-np.ones(n-1), diagonal, -np.ones(n-1)), (-1, 0, 1), format="csr")


def _path_eigenvalues(n):
    return 4. * np.sin(np.pi * np.arange(n) / (2. * n)) ** 2


class StructuredCapacitanceSolve:
    """SuperLU.solve-compatible real SPD solver, with one construction check.

    The numerical matrix must agree with the separable formula before use.
    Mutating family capacitance, state ordering or parameters afterward is
    unsupported, as for the original cached matrix/LU implementation.
    """
    def __init__(self, family):
        if family.structure not in ("R2", "R3"):
            raise ValueError("structured solve requires the unreduced R2/R3 circuit")
        self.rows, self.cols = family.rows, family.cols
        self.pads, self.driven = family.pads, family.driven
        self.size = family.size
        self.pad_free = np.delete(np.arange(self.pads), self.driven)
        free = np.delete(np.arange(family.full_size), self.driven)
        projection = sp.csr_matrix((np.ones(len(free)), (free, np.arange(len(free)))),
                                  shape=(family.full_size, len(free)))
        if (not np.array_equal(free, family.free)
                or (projection - family.projection).nnz):
            raise ValueError("structured solve cannot change the original free-state ordering")
        pad_matrix = (PAD_C * sp.eye(self.pads, format="csr")
                      + PAD_NEIGHBOUR_C * (sp.kron(_path_laplacian(self.rows), sp.eye(self.cols))
                                           + sp.kron(sp.eye(self.rows), _path_laplacian(self.cols))))
        source_base = SOURCE_C + (INPUT_C if family.structure == "R3" else 0.)
        source_coupling = family.parameters.adjacent_source_capacitance_farad
        source_matrix = (source_base * sp.eye(self.cols, format="csr")
                         + source_coupling * _path_laplacian(self.cols))
        extra = ([BUFFER_C] * self.cols + [FILTER_C] * self.cols
                 if family.structure == "R3" else [])
        extra += [HOLD_CAPACITANCE_FARAD] * family.adc_count
        self.extra_diagonal = np.asarray(extra, dtype=float)
        expected = sp.block_diag((pad_matrix.tocsr()[self.pad_free][:, self.pad_free],
                                 source_matrix, sp.diags(self.extra_diagonal)), format="csc")
        difference = (expected - family.capacitance).tocoo()
        # Assembly may sum equal branch terms in a different order. This
        # bound is at roundoff scale for the smallest nonzero circuit entry;
        # it does not authorize omitting any nonzero physical branch.
        if difference.nnz:
            scale = np.asarray(abs(expected[difference.row, difference.col])).ravel()
            if (np.any(scale == 0.) or np.any(np.abs(difference.data) > 8*np.finfo(float).eps*scale)):
                raise ValueError("capacitance is not the declared separable full-state matrix")
        self.pad_eigenvalues = (PAD_C + PAD_NEIGHBOUR_C *
            (_path_eigenvalues(self.rows)[:, None] + _path_eigenvalues(self.cols)[None, :]))
        self.source_eigenvalues = source_base + source_coupling * _path_eigenvalues(self.cols)
        self.source_is_diagonal = source_coupling == 0.
        impulse = np.zeros((self.rows, self.cols, 1))
        impulse.reshape(self.pads, 1)[self.driven] = 1.
        self.ground_column = self._pad_inverse(impulse).reshape(self.pads)
        self.ground_diagonal = self.ground_column[self.driven]
        if not np.isfinite(self.ground_diagonal) or self.ground_diagonal <= 0.:
            raise ArithmeticError("invalid driven-pad Dirichlet denominator")

    def _pad_inverse(self, rhs):
        modes = fft.dctn(rhs, type=2, norm="ortho", axes=(0, 1), workers=1)
        modes /= self.pad_eigenvalues[:, :, None]
        return fft.idctn(modes, type=2, norm="ortho", axes=(0, 1), workers=1, overwrite_x=True)

    def solve(self, rhs, trans="N"):
        if trans not in ("N", "T", "H"):
            raise ValueError("trans must be N, T or H")
        rhs = np.asarray(rhs)
        if np.iscomplexobj(rhs):
            raise TypeError("the original real capacitance factor accepts real right-hand sides only")
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.size:
            raise ValueError("right-hand side must name every free physical state")
        vector = rhs.ndim == 1
        columns = 1 if vector else rhs.shape[1]
        values = np.asarray(rhs, dtype=np.float64).reshape(self.size, columns)
        result = np.empty_like(values)
        pad_rhs = np.zeros((self.pads, columns))
        pad_rhs[self.pad_free] = values[:self.pads-1]
        pads = self._pad_inverse(pad_rhs.reshape(self.rows, self.cols, columns)).reshape(self.pads, columns)
        pads -= self.ground_column[:, None] * (pads[self.driven] / self.ground_diagonal)
        result[:self.pads-1] = pads[self.pad_free]
        first, last = self.pads-1, self.pads-1+self.cols
        if self.source_is_diagonal:
            result[first:last] = values[first:last] / self.source_eigenvalues[:, None]
        else:
            modes = fft.dct(values[first:last], type=2, norm="ortho", axis=0, workers=1)
            modes /= self.source_eigenvalues[:, None]
            result[first:last] = fft.idct(modes, type=2, norm="ortho", axis=0, workers=1, overwrite_x=True)
        result[last:] = values[last:] / self.extra_diagonal[:, None]
        return result.ravel() if vector else result


class StructuredAirFamily(AirFamily):
    """Original exponential action, with the exact capacitance solve replaced."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_factor = StructuredCapacitanceSolve(self)


class StructuredCachedAirFamily(CachedAirFamily):
    """Scalar-plan reuse plus the exact separable capacitance inverse."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap_factor = StructuredCapacitanceSolve(self)
