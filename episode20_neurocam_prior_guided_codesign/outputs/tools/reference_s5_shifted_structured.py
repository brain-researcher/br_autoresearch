"""Isolated exact (z*C+h*G) inverse prototype; no exponential integration.

All physical states are recovered. The pad block is separated by orthonormal
DCTs, with a rank-one selected-row update per column mode. The R2 source/hold
Schur complement is solved densely; R3 retains its one-way controlled source
and recovers the downstream buffer/filter/hold stars without symmetrizing it.
The driven pad is removed by a principal-inverse Dirichlet correction.
Complex algebra uses ordinary transpose, never conjugate transpose.
"""
import warnings

import numpy as np
import scipy.fft as fft
import scipy.linalg as la

from reference_s5_family import (PAD_C, PAD_LEAK, PAD_NEIGHBOUR_C, SOURCE_C,
    TFT_ON, TFT_OFF, INPUT_C, INPUT_G, BUFFER_C, BUFFER_G, FILTER_C,
    OUTPUT_G, FILTER_G, CONTROLLED_GAIN, HOLD_CAPACITANCE_FARAD,
    TRACK_CONDUCTANCE_SIEMENS, OFF_CONDUCTANCE_SIEMENS,
    HOLD_LEAK_CONDUCTANCE_SIEMENS)


def _eigenvalues(n):
    return 4*np.sin(np.pi*np.arange(n)/(2*n))**2


class StructuredShiftedInverse:
    """Only normal solves on the full64 R2/R3 circuit, for a single exact h.

    This prototype is not imported by production. No pole approximation,
    duration rounding, model reduction, or physics-level stopping is present.
    """
    def __init__(self, family, pole, duration, row, slot):
        if (family.structure not in ("R2", "R3")
                or (family.rows, family.cols, family.adc_count) != (64, 64, 24)):
            raise ValueError("isolated prototype requires full64 R2/R3 and24 holds")
        z, h = complex(pole), float(duration)
        if (not np.isfinite(z) or not np.isfinite(h) or h <= 0 or z == 0
                or not 0 <= row < 64 or not 0 <= slot < 4):
            raise ValueError("finite nonzero pole, positive actual h and valid action required")
        self.structure, self.size, self.full_size = family.structure, family.size, family.full_size
        self.driven, self.pads = int(family.driven), 4096
        self.free = np.delete(np.arange(self.full_size), self.driven)
        if not np.array_equal(self.free, family.free) or self.size != self.full_size-1:
            raise ValueError("original full physical state order required")
        self.check = family.check
        self.check()
        self.group = np.arange(64) % 24
        self.groups = np.eye(24)[self.group]  # source x hold incidence
        self.column_dct = fft.dct(np.eye(64), type=2, norm="ortho", axis=0, workers=1)
        eigen = _eigenvalues(64)
        self.pad_inverse_diagonal = 1/(z*(PAD_C+PAD_NEIGHBOUR_C*
            (eigen[:, None]+eigen[None, :]))+h*(PAD_LEAK+TFT_OFF))
        impulse = np.zeros(64)
        impulse[row] = 1.
        self.row_mode = fft.dct(impulse, type=2, norm="ortho", workers=1)
        delta = h*(TFT_ON-TFT_OFF)
        denominator = 1+delta*np.sum(self.row_mode[:, None]**2*self.pad_inverse_diagonal, axis=0)
        self._require_denominator(denominator, "selected-row correction")
        self.row_factor = delta/denominator
        conductance = np.full(64, TFT_OFF)
        conductance[row] = TFT_ON
        self.coupling_mode = h*fft.dct(conductance, type=2, norm="ortho", workers=1)
        self.pad_source_lift = self._pad_inverse_modes(np.broadcast_to(
            self.coupling_mode[:, None, None], (64, 64, 1)))[:, :, 0]
        source_base = SOURCE_C+(INPUT_C if self.structure == "R3" else 0.)
        source_load = family.parameters.source_load_siemens+(INPUT_G if self.structure == "R3" else 0.)
        source_diagonal = z*(source_base+family.parameters.adjacent_source_capacitance_farad*eigen)
        # Includes the TFT at the subsequently grounded driven pad.
        source_diagonal += h*(source_load+TFT_ON+63*TFT_OFF)
        self.source_modes = source_diagonal-np.sum(self.coupling_mode[:, None]*self.pad_source_lift, axis=0)
        self._require_denominator(self.source_modes, "source Schur modes")
        tft = np.where(np.arange(64)//24 == slot,
                       TRACK_CONDUCTANCE_SIEMENS, OFF_CONDUCTANCE_SIEMENS)
        self.track = h*tft
        self.hold_diagonal = (np.full(24, z*HOLD_CAPACITANCE_FARAD+h*HOLD_LEAK_CONDUCTANCE_SIEMENS)
                              + self.groups.T @ self.track)
        self.hold_diagonal[0] += h*(TRACK_CONDUCTANCE_SIEMENS if slot == 3 else OFF_CONDUCTANCE_SIEMENS)
        self._require_denominator(self.hold_diagonal, "physical hold diagonal")
        if self.structure == "R2":
            self.hold_coupling = self.track[:, None]*self.groups
            small = self.column_dct.T @ (self.source_modes[:, None]*self.column_dct)
            small[np.diag_indices(64)] += self.track
            small -= (self.hold_coupling/self.hold_diagonal[None, :]) @ self.hold_coupling.T
            with warnings.catch_warnings():
                warnings.simplefilter("error", la.LinAlgWarning)
                self.source_factor = la.lu_factor(small, check_finite=True)
        else:
            self.buffer_diagonal = z*BUFFER_C+h*(BUFFER_G+OUTPUT_G)
            self.output_coupling = h*OUTPUT_G
            self.controlled_coupling = h*BUFFER_G*CONTROLLED_GAIN
            self._require_denominator(self.buffer_diagonal, "buffer diagonal")
            self.filter_diagonal = z*FILTER_C+h*(OUTPUT_G+FILTER_G)+self.track
            self.filter_diagonal -= self.output_coupling**2/self.buffer_diagonal
            self._require_denominator(self.filter_diagonal, "filter Schur diagonal")
            self.hold_schur = self.hold_diagonal-self.groups.T @ (self.track**2/self.filter_diagonal)
            self._require_denominator(self.hold_schur, "downstream hold Schur diagonal")
        # Principal submatrix identity is valid also for nonsymmetric R3.
        # It uses the right inverse column, and no conjugation or left solve.
        unit = np.zeros((self.full_size, 1), dtype=complex)
        unit[self.driven] = 1.
        self.ground_column = self._full_solve(unit)[:, 0]
        self.ground_diagonal = self.ground_column[self.driven]
        self._require_denominator(self.ground_diagonal, "driven-pad Dirichlet correction")
        self.check()

    @staticmethod
    def _require_denominator(value, name):
        if not np.all(np.isfinite(value)) or np.any(np.asarray(value) == 0):
            raise ArithmeticError("invalid "+name)

    def _pad_inverse_modes(self, modes):
        answer = self.pad_inverse_diagonal[:, :, None]*modes
        projection = np.sum(self.row_mode[:, None, None]*answer, axis=0)
        answer -= (self.pad_inverse_diagonal*self.row_mode[:, None])[:, :, None]*(self.row_factor[:, None]*projection)[None, :, :]
        return answer

    def _full_solve(self, rhs):
        columns = rhs.shape[1]
        modes = fft.dctn(rhs[:4096].reshape(64, 64, columns), type=2,
                         norm="ortho", axes=(0, 1), workers=1)
        pads = self._pad_inverse_modes(modes)
        source_rhs = fft.dct(rhs[4096:4160], type=2, norm="ortho", axis=0, workers=1)
        source_rhs += np.sum(self.coupling_mode[:, None, None]*pads, axis=0)
        output = np.empty_like(rhs)
        if self.structure == "R2":
            small_rhs = fft.idct(source_rhs, type=2, norm="ortho", axis=0, workers=1)
            small_rhs += self.hold_coupling @ (rhs[4160:]/self.hold_diagonal[:, None])
            sources = la.lu_solve(self.source_factor, small_rhs, check_finite=False)
            output[4160:] = (rhs[4160:]+self.hold_coupling.T @ sources)/self.hold_diagonal[:, None]
            source_modes = fft.dct(sources, type=2, norm="ortho", axis=0, workers=1)
        else:
            source_modes = source_rhs/self.source_modes[:, None]
            sources = fft.idct(source_modes, type=2, norm="ortho", axis=0, workers=1)
            buffer_rhs = rhs[4160:4224]+self.controlled_coupling*sources
            filter_rhs = rhs[4224:4288]+(self.output_coupling/self.buffer_diagonal)*buffer_rhs
            holds = (rhs[4288:]+self.groups.T @ (self.track[:, None]*filter_rhs/self.filter_diagonal[:, None]))/self.hold_schur[:, None]
            filters = (filter_rhs+self.track[:, None]*holds[self.group])/self.filter_diagonal[:, None]
            output[4160:4224] = (buffer_rhs+self.output_coupling*filters)/self.buffer_diagonal
            output[4224:4288], output[4288:] = filters, holds
        output[4096:4160] = sources
        pads += self.pad_source_lift[:, :, None]*source_modes[None, :, :]
        output[:4096] = fft.idctn(pads, type=2, norm="ortho", axes=(0, 1), workers=1).reshape(4096, columns)
        return output

    def solve(self, rhs):
        values = np.asarray(rhs, dtype=complex)
        if values.ndim not in (1, 2) or values.shape[0] != self.size:
            raise ValueError("right-hand side must retain every original free coordinate")
        vector = values.ndim == 1
        columns = 1 if vector else values.shape[1]
        padded = np.zeros((self.full_size, columns), dtype=complex)
        padded[self.free] = values.reshape(self.size, columns)
        full = self._full_solve(padded)
        full -= self.ground_column[:, None]*(full[self.driven]/self.ground_diagonal)
        answer = full[self.free]
        return answer[:, 0] if vector else answer

    def retained_numeric_bytes(self):
        total = 0
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                total += value.nbytes
            elif isinstance(value, tuple):
                total += sum(item.nbytes for item in value if isinstance(item, np.ndarray))
        return total
