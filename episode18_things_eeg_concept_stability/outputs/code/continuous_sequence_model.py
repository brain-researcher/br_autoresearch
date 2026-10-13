"""Sequence-local, matrix-free time expansion and direct joint EEG ridge fitting.

This numerical component performs no file access and establishes no reader
isolation. Its caller must supply permitted arrays inside coordinated compute,
the frozen physical-sequence/event join, training-only features and nuisance
terms, and the same scoring mask/sensor transform for all compared models.
It does not select preprocessing, a temporal basis, folds, tuning, or penalties.

For each sequence and term, X[t, (feature, basis)] is the sum over that
sequence's events of feature[event] * basis(sample_time[t] - event_time).
Times are explicit seconds in one common coordinate system per sequence;
onsets are never rounded to the target sampling grid. The supplied basis is
evaluated at those actual lags only within its declared closed support.
All nuisance and event columns enter one fit; no exported residual target is
created. Event ablation subtracts only the selected fitted term contributions,
on the complete supplied sequence, without refitting or changing its mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr


def _real_array(value, name: str) -> np.ndarray:
    result = np.asarray(value)
    if np.iscomplexobj(result):
        raise ValueError(f"{name} must be real-valued; complex components cannot be discarded")
    return np.asarray(result, dtype=np.float64)


def _finite_array(value, ndim: int, name: str) -> np.ndarray:
    result = _real_array(value, name).copy()
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class EventTerm:
    """One feature/added-slot family with an explicit, reusable temporal basis.

    temporal_basis(lags_seconds) returns [len(lags_seconds), basis_dimension].
    The exact same callable and column definitions must be used across the
    sequences in a fit. For the primary model, the caller supplies the selected
    registered basis with support (0.0, 0.8); no basis is chosen by this module.
    Event rows retain their caller's order, including simultaneous events.
    """

    onsets_seconds: np.ndarray
    event_features: np.ndarray
    temporal_basis: Callable[[np.ndarray], np.ndarray]
    basis_dimension: int
    support_seconds: tuple[float, float]


@dataclass(frozen=True)
class _PreparedTerm:
    features: np.ndarray
    kernels: tuple[tuple[int, int, np.ndarray], ...]
    basis_dimension: int
    column_slice: slice


class SequenceDesign:
    """One physical sequence, including every overlapping event contribution.

    sample_times_seconds may include samples later masked from fitting/scoring;
    event support is clipped to this sequence's supplied samples only. Events
    outside a trimmed sample range can still contribute inside it. Membership
    in the same physical sequence must come from the frozen source join.
    nuisance is [samples, nuisance_columns], including an explicit intercept
    if the caller's selected nuisance recipe requires one. Nothing is centered
    or residualized here. Dict insertion order defines event column order.
    """

    def __init__(self, sequence_id: str, sample_times_seconds, nuisance,
                 terms: Mapping[str, EventTerm]):
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("An explicit physical sequence ID is required")
        self.sequence_id = sequence_id
        self.sample_times = _finite_array(sample_times_seconds, 1, "sample times")
        if not len(self.sample_times) or np.any(np.diff(self.sample_times) <= 0):
            raise ValueError("Sample times must be nonempty and strictly increasing")
        self.nuisance = _finite_array(nuisance, 2, "nuisance")
        if self.nuisance.shape[0] != len(self.sample_times):
            raise ValueError("Nuisance rows must match this physical sequence")
        self.terms: dict[str, _PreparedTerm] = {}
        schema = []
        column = self.nuisance.shape[1]
        for name, term in terms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Each event term needs a nonempty name")
            onsets = _finite_array(term.onsets_seconds, 1, "event onsets")
            features = _finite_array(term.event_features, 2, "event features")
            if features.shape[0] != len(onsets):
                raise ValueError("Event feature rows/dimension do not match onsets")
            # Registered training PCA can have effective rank zero. Retain
            # this named family and its event rows with zero coefficient
            # columns; other terms/nuisance must still identify a joint model.
            if (isinstance(term.basis_dimension, bool)
                    or not isinstance(term.basis_dimension, (int, np.integer))
                    or term.basis_dimension < 1 or not callable(term.temporal_basis)):
                raise ValueError("Supply a callable basis and positive integer dimension")
            support = _finite_array(term.support_seconds, 1, "temporal support")
            if support.shape != (2,) or support[0] > support[1]:
                raise ValueError("Temporal support must be an ordered pair in seconds")
            kernels = []
            for onset in onsets:
                # Subtract actual times before applying support: no event-grid
                # rounding and no neighboring physical sequence's events.
                lags = self.sample_times - onset
                lo = int(np.searchsorted(lags, support[0], side="left"))
                hi = int(np.searchsorted(lags, support[1], side="right"))
                if hi > lo:
                    kernel = _finite_array(term.temporal_basis(lags[lo:hi]), 2,
                                           "evaluated temporal basis")
                    if kernel.shape != (hi - lo, term.basis_dimension):
                        raise ValueError("Temporal basis returned the wrong shape")
                else:
                    kernel = np.empty((0, term.basis_dimension), dtype=np.float64)
                    kernel.setflags(write=False)
                kernels.append((lo, hi, kernel))
            end = column + features.shape[1] * term.basis_dimension
            self.terms[name] = _PreparedTerm(features, tuple(kernels),
                                             term.basis_dimension, slice(column, end))
            schema.append((name, features.shape[1], term.basis_dimension,
                           tuple(support), term.temporal_basis))
            column = end
        if not column:
            raise ValueError("The joint model needs at least one explicit predictor")
        self.schema = (self.nuisance.shape[1], tuple(schema))
        self.shape = (len(self.sample_times), column)

    def _predict(self, coefficients):
        result = self.nuisance @ coefficients[:self.nuisance.shape[1]]
        for term in self.terms.values():
            amplitudes = term.features @ coefficients[term.column_slice].reshape(
                term.features.shape[1], term.basis_dimension)
            for event, (lo, hi, kernel) in enumerate(term.kernels):
                result[lo:hi] += kernel @ amplitudes[event]
        return result

    def _transpose(self, sample_vector):
        result = np.empty(self.shape[1], dtype=np.float64)
        result[:self.nuisance.shape[1]] = self.nuisance.T @ sample_vector
        for term in self.terms.values():
            event_products = np.empty((len(term.features), term.basis_dimension))
            for event, (lo, hi, kernel) in enumerate(term.kernels):
                event_products[event] = kernel.T @ sample_vector[lo:hi]
            result[term.column_slice] = (term.features.T @ event_products).ravel()
        return result


class JointSequenceDesign(LinearOperator):
    """Stack physical sequences vertically, sharing the declared coefficients.

    No event or nuisance sample ever flows between sequence blocks. A caller
    must construct fitting and held-out designs from their respective whole
    blocks; this operator has no data reader, split selector or access bypass.
    """

    def __init__(self, sequences: Sequence[SequenceDesign]):
        self.sequences = tuple(sequences)
        if not self.sequences:
            raise ValueError("Supply at least one physical sequence")
        ids = [part.sequence_id for part in self.sequences]
        if len(set(ids)) != len(ids):
            raise ValueError("A physical sequence may appear only once in a design")
        if any(part.schema != self.sequences[0].schema for part in self.sequences):
            raise ValueError("All sequences need identical term/basis column definitions")
        ends = np.cumsum([0] + [part.shape[0] for part in self.sequences])
        self.row_slices = tuple(slice(int(lo), int(hi))
                                for lo, hi in zip(ends[:-1], ends[1:]))
        super().__init__(dtype=np.dtype(np.float64),
                         shape=(int(ends[-1]), self.sequences[0].shape[1]))

    def _matvec(self, coefficients):
        values = _real_array(coefficients, "coefficients").reshape(-1)
        return np.concatenate([part._predict(values) for part in self.sequences])

    def _rmatvec(self, sample_vector):
        values = _real_array(sample_vector, "sample vector").reshape(-1)
        result = np.zeros(self.shape[1], dtype=np.float64)
        for part, rows in zip(self.sequences, self.row_slices):
            result += part._transpose(values[rows])
        return result

    def event_contribution(self, coefficients, *, term_name: str,
                           event_masks: Mapping[str, np.ndarray]):
        """Predict only selected events of one fitted term over full support.

        Coefficients may be [columns] or [columns, sensor_contrasts]. A Boolean
        mask in the original event-row order is required for every sequence.
        The caller chooses distant held-out events using the frozen EEG-blind
        distance panel; neither EEG nor outcomes select events in this method.
        """
        coefficients = _real_array(coefficients, "coefficients")
        if (coefficients.ndim not in (1, 2) or coefficients.shape[0] != self.shape[1]
                or not np.isfinite(coefficients).all()):
            raise ValueError("Supply the complete finite fitted coefficient array")
        if set(event_masks) != {part.sequence_id for part in self.sequences}:
            raise ValueError("Supply one event mask per physical sequence")
        single = coefficients.ndim == 1
        matrix = coefficients[:, None] if single else coefficients
        result = np.zeros((self.shape[0], matrix.shape[1]), dtype=np.float64)
        for part, rows in zip(self.sequences, self.row_slices):
            if term_name not in part.terms:
                raise ValueError("Unknown fitted event term")
            term = part.terms[term_name]
            selected = np.asarray(event_masks[part.sequence_id])
            if selected.dtype != np.bool_ or selected.shape != (len(term.features),):
                raise ValueError("Event masks must be Boolean and preserve event rows")
            beta = matrix[term.column_slice].reshape(
                term.features.shape[1], term.basis_dimension, matrix.shape[1])
            for event in np.flatnonzero(selected):
                lo, hi, kernel = term.kernels[event]
                amplitudes = np.tensordot(term.features[event], beta, axes=(0, 0))
                result[rows.start + lo:rows.start + hi] += kernel @ amplitudes
        return result[:, 0] if single else result

    def ablate_events(self, coefficients, *, term_name: str,
                      event_masks: Mapping[str, np.ndarray]):
        """Subtract selected fitted contributions; never refit or change masks."""
        removed = self.event_contribution(coefficients, term_name=term_name,
                                          event_masks=event_masks)
        return self @ np.asarray(coefficients) - removed


@dataclass(frozen=True)
class JointRidgeFit:
    coefficients: np.ndarray
    stop_codes: tuple[int, ...]
    iterations: tuple[int, ...]


def fit_joint_ridge(design: JointSequenceDesign, targets, *, sample_weights,
                    penalties, atol: float, btol: float, conlim: float,
                    maxiter: int) -> JointRidgeFit:
    """Fit all supplied nuisance/event predictors directly to the same targets.

    Minimize sum_t w[t] * ||Y[t] - X[t] B||^2 + sum_j lambda[j] * ||B[j]||^2
    using the augmented least-squares operator, without forming X or X'X.
    weights and penalties are explicit; zero weights exclude fixed samples
    and zero penalties allow unpenalized nuisance terms. No mean removal,
    nuisance-first residual fit, penalty normalization or tuning is implicit.
    Multiple sensor contrasts are solved serially with the same operator.
    Unconverged/condition-limited fits raise, rather than returning a result.
    """
    y = _real_array(targets, "targets")
    single = y.ndim == 1
    if single:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] != design.shape[0] or y.shape[1] < 1:
        raise ValueError("Targets must match all supplied physical-sequence rows")
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite targets; sample handling must be defined upstream")
    weights = _finite_array(sample_weights, 1, "sample weights")
    penalty = _finite_array(penalties, 1, "column penalties")
    if weights.shape != (design.shape[0],) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Supply nonnegative row weights with positive fitting support")
    if penalty.shape != (design.shape[1],) or np.any(penalty < 0):
        raise ValueError("Supply one nonnegative ridge penalty per joint column")
    if (not np.isfinite([atol, btol, conlim]).all() or atol <= 0 or btol <= 0
            or conlim <= 1 or isinstance(maxiter, bool)
            or not isinstance(maxiter, (int, np.integer)) or maxiter < 1):
        raise ValueError("Supply positive solver tolerances and a finite iteration bound")
    root_w, root_penalty = np.sqrt(weights), np.sqrt(penalty)
    rows, columns = design.shape

    def forward(beta):
        beta = np.asarray(beta).reshape(-1)
        return np.concatenate((root_w * design.matvec(beta), root_penalty * beta))

    def transpose(values):
        values = np.asarray(values).reshape(-1)
        return design.rmatvec(root_w * values[:rows]) + root_penalty * values[rows:]

    augmented = LinearOperator((rows + columns, columns), matvec=forward,
                                rmatvec=transpose, dtype=np.float64)
    coefficients, stop_codes, iterations = [], [], []
    for target in y.T:
        rhs = np.concatenate((root_w * target, np.zeros(columns)))
        solution = lsmr(augmented, rhs, atol=atol, btol=btol, conlim=conlim,
                        maxiter=int(maxiter))
        beta, stop, count = solution[:3]
        if stop not in (0, 1, 2, 4, 5) or not np.isfinite(beta).all():
            raise RuntimeError(f"Joint ridge did not converge: stop={stop}, iterations={count}")
        coefficients.append(beta)
        stop_codes.append(int(stop))
        iterations.append(int(count))
    matrix = np.column_stack(coefficients)
    return JointRidgeFit(matrix[:, 0] if single else matrix,
                         tuple(stop_codes), tuple(iterations))
