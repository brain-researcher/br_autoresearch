"""Bounded direct solver for EP18's unchanged, small core-reference model.

The full categorical models are deliberately outside this implementation.
SequenceRows preserves every original physical-sequence row, actual onset,
overlapping event, nuisance column and weight. Sufficient statistics are
formed once for one training state and reused across its thirteen penalties.
No rank truncation, jitter, penalty rescaling or held-out information is used.
Every solve checks its backward residual; a failed factorization stops.
The caller chooses this backend explicitly within one process. Historical
matrix-free functions and all running search jobs remain unchanged.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

import continuous_fold_runner as folds
from continuous_sequence_model import JointSequenceDesign
from streamed_slot_capacity import SequenceRows


@dataclass(frozen=True)
class DirectRidgeFit:
    coefficients: np.ndarray
    stop_codes: tuple
    iterations: tuple
    backend: str = "weighted_gram_cholesky_v1"


class DirectCoreRidge:
    def __init__(self, *, check, chunk_rows=2048, maximum_columns=4096):
        self.check, self.chunk_rows, self.maximum_columns = check, chunk_rows, maximum_columns
        if not callable(check) or type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("Explicit bounded row blocks and budget callback required")
        self._identity = None
        self._gram = self._rhs = None
        self.stats = {"training_statistics_built": 0, "training_statistics_reused": 0,
                      "solves": 0, "statistics_seconds": 0., "solve_seconds": 0.,
                      "maximum_relative_backward_residual": 0.}

    def clear(self):
        self._identity = self._gram = self._rhs = None

    def _statistics(self, design, targets, weights):
        identity = (design, targets, weights)
        if self._identity is not None and all(a is b for a, b in zip(identity, self._identity)):
            self.stats["training_statistics_reused"] += 1
            return self._gram, self._rhs
        self.clear()
        self.check()
        n, p = design.shape
        if not isinstance(design, JointSequenceDesign) or p > self.maximum_columns:
            raise ValueError("Direct backend is bounded to the complete small core design")
        if (targets.ndim != 2 or targets.shape != (n, 61) or weights.shape != (n,)
                or not np.isfinite(targets).all() or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.any(weights > 0)):
            raise ValueError("Complete finite 61-contrast targets and fixed weights required")
        started = time.monotonic()
        gram, rhs = np.zeros((p, p)), np.zeros((p, 61))
        for first, last, block in SequenceRows(design, 0, p).blocks(
                chunk_rows=self.chunk_rows, budget_check=self.check):
            self.check()
            root = np.sqrt(weights[first:last])
            block *= root[:, None]
            gram += block.T @ block
            rhs += block.T @ (root[:, None] * targets[first:last])
        self.check()
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
            raise FloatingPointError("Nonfinite weighted core sufficient statistics")
        self._identity, self._gram, self._rhs = identity, gram, rhs
        self.stats["training_statistics_built"] += 1
        self.stats["statistics_seconds"] += time.monotonic() - started
        return gram, rhs

    def fit(self, design, targets, weights, penalties, limits, check):
        check()
        gram, rhs = self._statistics(design, targets, weights)
        penalty = np.asarray(penalties, dtype=np.float64)
        if penalty.shape != (gram.shape[0],) or not np.isfinite(penalty).all() or np.any(penalty < 0):
            raise ValueError("One unchanged nonnegative penalty per original column required")
        started = time.monotonic()
        matrix = gram.copy()
        matrix.flat[::len(matrix)+1] += penalty
        # Cholesky is permitted only for the unmodified positive-definite
        # weighted objective. There is no ridge jitter or pseudoinverse rescue.
        factor = cho_factor(matrix, lower=True, check_finite=False)
        beta = cho_solve(factor, rhs, check_finite=False)
        residual = matrix @ beta - rhs
        scale = np.linalg.norm(matrix, ord=np.inf) * np.max(np.abs(beta), axis=0) + np.max(np.abs(rhs), axis=0)
        absolute = np.max(np.abs(residual), axis=0)
        relative = np.divide(absolute, scale, out=np.zeros_like(absolute), where=scale > 0)
        if (not np.isfinite(beta).all() or np.any((scale == 0) & (absolute != 0))
                or not np.isfinite(relative).all() or np.max(relative) > min(limits.atol, limits.btol)):
            raise ArithmeticError("Direct core solve failed the original tolerance residual check")
        check()
        self.stats["solves"] += 1
        self.stats["solve_seconds"] += time.monotonic() - started
        self.stats["maximum_relative_backward_residual"] = max(
            self.stats["maximum_relative_backward_residual"], float(np.max(relative)))
        beta.setflags(write=False)
        # These are explicit direct-solver labels, never invented LSMR codes.
        return DirectRidgeFit(beta, ("direct_backward_residual_pass",)*61, (0,)*61)

    def predict(self, design, coefficients):
        values = np.asarray(coefficients, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != design.shape[1]:
            raise ValueError("Complete multiresponse coefficient matrix required")
        result = np.empty((design.shape[0], values.shape[1]))
        for first, last, block in SequenceRows(design, 0, design.shape[1]).blocks(
                chunk_rows=self.chunk_rows, budget_check=self.check):
            result[first:last] = block @ values
        return result

    @contextmanager
    def selected(self):
        """Explicit single-process injection into existing core algebra only."""
        old_fit = folds._fit
        own_matmat = JointSequenceDesign.__dict__.get("_matmat")
        folds._fit = self.fit
        backend = self
        JointSequenceDesign._matmat = lambda design, values: backend.predict(design, values)
        try:
            yield self
        finally:
            folds._fit = old_fit
            if own_matmat is None:
                del JointSequenceDesign._matmat
            else:
                JointSequenceDesign._matmat = own_matmat
            self.clear()
