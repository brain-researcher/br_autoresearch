"""Sparse passive R2 integration on the fixed 4096-pad/64-source topology.

No device constants, DAQ configuration or contact-CPE approximation is supplied.
The caller gives all passive branches, drives, initial states, charge injection
and numerical controls. Off-gate pads and unread sources remain physical nodes.
This is an analog integration layer, not a candidate scorer or reference fit.
"""

from dataclasses import dataclass
import math
from typing import ClassVar

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, expm_multiply, splu

from coupled_rc import ChargeTransition, NetworkInterval


def _array(name, value, shape):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _edges(name, value, size):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=float, copy=True)
    if result.size == 0:
        result = result.reshape(0, 3)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite (node_i, node_j, value) rows")
    endpoints = result[:, :2]
    if (np.any(endpoints != np.floor(endpoints)) or np.any(endpoints < 0)
            or np.any(endpoints >= size) or np.any(endpoints[:, 0] == endpoints[:, 1])
            or np.any(result[:, 2] < 0)):
        raise ValueError(f"{name} needs distinct in-range integer nodes and nonnegative values")
    return result


def passive_matrix(*, shunts, edges, positive_shunts):
    """Assemble diag(shunts) + sum w(e_i-e_j)(e_i-e_j)^T in sparse form.

    Nonnegative physical branches establish passivity by construction. Strictly
    positive reference capacitances make C positive definite, without a dense
    eigenvalue test, hidden epsilon, clipping or physical-parameter repair.
    Duplicate edges are explicit parallel branches and are added once each.
    """
    size = len(shunts)
    shunts = _array("shunts", shunts, (size,))
    if np.any(shunts < 0) or (positive_shunts and np.any(shunts <= 0)):
        raise ValueError("shunts must be nonnegative (strictly positive for capacitance)")
    edges = _edges("edges", edges, size)
    first, second = edges[:, 0].astype(int), edges[:, 1].astype(int)
    weight = edges[:, 2]
    rows = np.concatenate((first, second, first, second))
    cols = np.concatenate((first, second, second, first))
    values = np.concatenate((weight, weight, -weight, -weight))
    matrix = (diags(shunts, format="csr") + coo_matrix((values, (rows, cols)), shape=(size, size))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("assembled matrix must be finite")
    return matrix


def _tuple(array):
    return tuple(float(value) for value in array)


def _charge_energy(capacitance, volts):
    charge = capacitance @ volts
    energy = float(0.5 * (volts @ charge))
    if not np.all(np.isfinite(charge)) or not math.isfinite(energy):
        raise ValueError("charge and stored energy must be finite")
    return charge, energy


@dataclass(frozen=True)
class SolverControls:
    """Explicit numerical controls, not fitted or scientific acceptance margins.

    Partition disagreement estimates numerical consistency, not a certified
    error bound. max_operator_products bounds propagation-operator applications
    across the attempted interval, including adjoints/norm estimation. Fixed
    boundary/drive solves and sparse factorization are outside that counter.
    """
    max_partition_refinements: ClassVar[int] = 8
    relative_tolerance: float
    voltage_atol: float
    charge_atol_coulomb: float
    max_refinements: int
    max_operator_products: int

    def __post_init__(self):
        if any(np.iscomplexobj(value) for value in (self.relative_tolerance,
                                                    self.voltage_atol, self.charge_atol_coulomb)):
            raise ValueError("numerical tolerances must be real")
        if (not math.isfinite(self.relative_tolerance) or self.relative_tolerance < 0
                or not math.isfinite(self.voltage_atol) or self.voltage_atol <= 0
                or not math.isfinite(self.charge_atol_coulomb) or self.charge_atol_coulomb <= 0):
            raise ValueError("numerical tolerances must be finite, with positive absolute tolerances")
        for name, value in (("max_refinements", self.max_refinements),
                            ("max_operator_products", self.max_operator_products)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_refinements > self.max_partition_refinements:
            raise ValueError(f"this bounded solver permits at most {self.max_partition_refinements} partition refinements")


@dataclass(frozen=True)
class SparseChargeTransition:
    transition: ChargeTransition
    maximum_scaled_charge_residual: float


@dataclass(frozen=True)
class SparseInterval:
    boundary: SparseChargeTransition
    observation: NetworkInterval
    estimated_endpoint_error_volts: float
    estimated_mean_error_volts: float
    maximum_scaled_partition_error: float
    subintervals: int
    operator_products: int


class SolverConvergenceError(RuntimeError):
    """A declared numerical/work limit failed; persistent state is unchanged."""


class PassiveStarConductance(LinearOperator):
    """Positive resistor stars with massless source junctions eliminated exactly.

    Each explicit group is (node_indices, branch_siemens, series_ohm). A
    prescribed voltage port drives the star junction through series_ohm;
    zero resistance is the exact ideal-port limit. For h >= 0 and H=sum(h),
    G=diag(h)-R*h*h'/(1+R*H), b=h/(1+R*H). All dynamic nodes are retained.
    Products/storage are linear in the number of star terms, not its square.
    This class accepts physical stars, not arbitrary conductance operators.
    """

    def __init__(self, *, state_count, groups):
        if isinstance(state_count, bool) or not isinstance(state_count, int) or state_count < 1:
            raise ValueError("state_count must be a positive integer")
        groups = tuple(groups)
        if not groups:
            raise ValueError("at least one explicit source star is required")
        nodes, weights, group_ids, resistances, references = [], [], [], [], []
        for group_id, (indices, conductances, resistance) in enumerate(groups):
            indices = tuple(indices)
            if (not indices or any(isinstance(i, bool) or not isinstance(i, int)
                                   or not 0 <= i < state_count for i in indices)
                    or len(set(indices)) != len(indices)):
                raise ValueError("each star needs distinct in-range dynamic nodes")
            h = _array("star branch conductance", conductances, (len(indices),))
            if np.any(h < 0) or np.iscomplexobj(resistance):
                raise ValueError("star conductances and series resistance must be nonnegative and real")
            resistance = float(resistance)
            if not math.isfinite(resistance) or resistance < 0:
                raise ValueError("star series resistance must be finite and nonnegative")
            nodes.extend(indices)
            weights.extend(h)
            group_ids.extend([group_id] * len(indices))
            resistances.append(resistance)
            references.append(indices[0])
        self._nodes = np.array(nodes, dtype=np.int64)
        self._h = np.array(weights, dtype=float)
        self._group = np.array(group_ids, dtype=np.int64)
        self._reference = np.array(references, dtype=np.int64)
        self._totals = np.bincount(self._group, weights=self._h, minlength=len(groups))
        product = np.array(resistances) * self._totals
        if not np.all(np.isfinite(product)) or not np.all(np.isfinite(self._totals)):
            raise ValueError("star totals and R*H must be representable")
        self._mix = product / (1 + product)
        self._b = self._h / (1 + product[self._group])
        fraction = np.divide(self._h, self._totals[self._group], out=np.zeros_like(self._h),
                             where=self._totals[self._group] > 0)
        self._diagonal = np.bincount(self._nodes,
                                    weights=self._b + self._h * self._mix[self._group] * (1 - fraction),
                                    minlength=state_count)
        if not np.all(np.isfinite(self._diagonal)):
            raise ValueError("star diagonal must be finite")
        for value in (self._nodes, self._h, self._group, self._reference, self._totals,
                      self._mix, self._b, self._diagonal):
            value.flags.writeable = False
        super().__init__(dtype=np.dtype(float), shape=(state_count, state_count))

    @property
    def port_count(self):
        return len(self._reference)

    @property
    def term_count(self):
        return len(self._nodes)

    @property
    def numeric_storage_bytes(self):
        return sum(value.nbytes for value in (self._nodes, self._h, self._group, self._reference,
                                              self._totals, self._mix, self._b, self._diagonal))

    def diagonal(self):
        return self._diagonal.copy()

    def _matmat(self, values):
        # Center within each star to avoid subtracting large equal common-mode
        # terms when R*H is large. This is the exact same Schur operator.
        local = values[self._nodes]
        differences = local - values[self._reference[self._group]]
        weighted_sum = np.zeros((self.port_count, values.shape[1]))
        np.add.at(weighted_sum, self._group, self._h[:, None] * differences)
        weighted_mean = np.divide(weighted_sum, self._totals[:, None],
                                  out=np.zeros_like(weighted_sum), where=self._totals[:, None] > 0)
        terms = (self._b[:, None] * local + self._h[:, None] * self._mix[self._group, None]
                 * (differences - weighted_mean[self._group]))
        result = np.zeros_like(values, dtype=float)
        np.add.at(result, self._nodes, terms)
        return result

    def _matvec(self, values):
        return self._matmat(np.asarray(values).reshape(-1, 1)).ravel()

    def _rmatmat(self, values):
        return self._matmat(values)

    def _rmatvec(self, values):
        return self._matvec(values)

    def drive_current(self, *, port_volts):
        ports = _array("port_volts", port_volts, (self.port_count,))
        current = np.bincount(self._nodes, weights=self._b * ports[self._group], minlength=self.shape[0])
        if not np.all(np.isfinite(current)):
            raise ValueError("source-star drive current must be finite")
        return current

    def source_work(self, *, port_volts, mean_state_volts, duration_seconds):
        """Work delivered by physical voltage ports, including series paths.

        This differs from the reduced nodal Norton-current work. Constant
        voltages and interval-mean states make this integral exact for the
        represented resistor network; no complete device power is implied.
        """
        ports = _array("port_volts", port_volts, (self.port_count,))
        mean = _array("mean_state_volts", mean_state_volts, (self.shape[0],))
        if np.iscomplexobj(duration_seconds) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite, real and positive")
        currents = np.bincount(self._group,
                               weights=self._b * (ports[self._group] - mean[self._nodes]),
                               minlength=self.port_count)
        work = float(duration_seconds * (ports @ currents))
        if not math.isfinite(work):
            raise ValueError("physical source work must be finite")
        return work


class _ConductanceWithStars(LinearOperator):
    def __init__(self, base, stars):
        self.base, self.stars = base, stars
        super().__init__(dtype=np.dtype(float), shape=base.shape)

    def diagonal(self):
        return self.base.diagonal() + self.stars.diagonal()

    def _matmat(self, values):
        return self.base @ values + self.stars @ values

    def _matvec(self, values):
        return self.base @ values + self.stars @ values

    def _rmatmat(self, values):
        return self._matmat(values)

    def _rmatvec(self, values):
        return self._matvec(values)


class _WorkCounter:
    def __init__(self, limit):
        self.limit = limit
        self.products = 0

    def add(self, columns):
        self.products += columns
        if self.products > self.limit:
            raise SolverConvergenceError("matrix-free operator-product limit exceeded")


class _IntervalOperator(LinearOperator):
    def __init__(self, factor, conductance, drive_derivative, duration, counter):
        self.factor = factor
        self.conductance = conductance
        self.drive = drive_derivative
        self.duration = duration
        self.counter = counter
        self.size = conductance.shape[0]
        super().__init__(dtype=np.dtype(float), shape=(2 * self.size + 1, 2 * self.size + 1))

    def _matmat(self, values):
        self.counter.add(values.shape[1])
        size = self.size
        result = np.zeros_like(values, dtype=float)
        result[:size] = (-self.factor.solve(self.conductance @ values[:size])
                         + self.drive[:, None] * values[size]) * self.duration
        result[size + 1:] = values[:size]
        return result

    def _rmatmat(self, values):
        self.counter.add(values.shape[1])
        size = self.size
        result = np.zeros_like(values, dtype=float)
        result[:size] = (-self.conductance.T @ self.factor.solve(values[:size], trans="T")) * self.duration
        result[:size] += values[size + 1:]
        result[size] = (self.drive @ values[:size]) * self.duration
        return result

    def _matvec(self, values):
        return self._matmat(np.asarray(values).reshape(-1, 1)).ravel()

    def _rmatvec(self, values):
        return self._rmatmat(np.asarray(values).reshape(-1, 1)).ravel()


class SparsePassiveRCNetwork:
    """Persistent sparse network with passive branches and retained state basis.

    Reference capacitance is nonnegative. Every connected component of positive
    mutual capacitors must contain a positive reference capacitor. This makes
    C positive definite while allowing ungrounded internal series-RC states.
    Mutual capacitors and resistor edges are nonnegative. Ideal constraints and
    active devices require explicit separate modeling.
    No dense C/G, inverse or exponential is formed; sparse LU can still incur
    fill-in for a caller's graph. Work limits do not bound that factorization.
    """

    def __init__(self, *, state_ids, initial_volts, capacitance_to_reference_farad,
                 mutual_capacitance_edges):
        if isinstance(state_ids, (str, bytes)):
            raise ValueError("state_ids must be a sequence of names")
        self._ids = tuple(state_ids)
        if (not self._ids or any(not isinstance(value, str) or not value for value in self._ids)
                or len(set(self._ids)) != len(self._ids)):
            raise ValueError("state_ids must be distinct nonempty names")
        self._size = len(self._ids)
        self._volts = _array("initial_volts", initial_volts, (self._size,))
        self._capacitance = self._storage(capacitance_to_reference_farad, mutual_capacitance_edges)
        self._factor = splu(self._capacitance.tocsc())
        self._pending_preview = None
        _charge_energy(self._capacitance, self._volts)

    def _storage(self, shunts, edges):
        shunts = _array("capacitance_to_reference_farad", shunts, (self._size,))
        capacitance = passive_matrix(shunts=shunts, edges=edges, positive_shunts=False)
        if np.any(shunts == 0):
            # Off-diagonal nonzeros are exactly the positive capacitor graph.
            # v'C v = sum C_i v_i^2 + sum C_ij (v_i-v_j)^2 is strictly positive
            # iff each connected component has a positive reference capacitor.
            count, labels = connected_components(capacitance, directed=False)
            grounded = np.bincount(labels, weights=(shunts > 0), minlength=count)
            if np.any(grounded == 0):
                raise ValueError("every capacitor component needs a positive reference capacitor")
        return capacitance

    @property
    def state_ids(self):
        return self._ids

    @property
    def volts(self):
        return _tuple(self._volts)

    @property
    def capacitance_nnz(self):
        return self._capacitance.nnz

    @property
    def numeric_storage_bytes(self):
        matrices = (self._capacitance, self._factor.L, self._factor.U)
        return (self._volts.nbytes + self._factor.perm_r.nbytes + self._factor.perm_c.nbytes
                + sum(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes for m in matrices))

    def _transition(self, capacitance, factor, injected, controls):
        before_charge, before_energy = _charge_energy(self._capacitance, self._volts)
        target_charge = before_charge + injected
        after = (self._volts.copy() if capacitance is self._capacitance and not np.any(injected)
                 else factor.solve(target_charge))
        after_charge, after_energy = _charge_energy(capacitance, after)
        denominator = controls.charge_atol_coulomb + controls.relative_tolerance * np.maximum(
            np.abs(target_charge), np.abs(after_charge))
        residual = float(np.max(np.abs(after_charge - target_charge) / denominator))
        if not math.isfinite(residual) or residual > 1:
            raise SolverConvergenceError("capacitance solve exceeds declared charge residual tolerance")
        transition = ChargeTransition(self.volts, _tuple(after), _tuple(before_charge),
                                      _tuple(injected), _tuple(after_charge), before_energy,
                                      after_energy, after_energy - before_energy)
        return after, SparseChargeTransition(transition, residual)

    def switch_capacitance(self, *, capacitance_to_reference_farad, mutual_capacitance_edges,
                           injected_charge_coulomb, controls):
        """Conserve nodal charge in the same basis; commit only after residual check.

        Stored-energy change is not complete switch work or power. No nodes are
        dropped, renamed or reset when a row/bank is inactive.
        """
        capacitance = self._storage(capacitance_to_reference_farad, mutual_capacitance_edges)
        injected = _array("injected_charge_coulomb", injected_charge_coulomb, (self._size,))
        factor = splu(capacitance.tocsc())
        after, transition = self._transition(capacitance, factor, injected, controls)
        self._capacitance, self._factor, self._volts = capacitance, factor, after
        self._pending_preview = None
        return transition

    def advance(self, *, duration_seconds, conductance_to_reference_siemens,
                resistor_edges, drive_current_amp, observation_matrix,
                feedthrough_volts, injected_charge_coulomb, controls,
                passive_stars=None, commit=True):
        """Propagate a constant-input interval and return mean-aperture observations.

        drive_current_amp is the explicit nodal Bv; feedthrough_volts is Dv.
        No source, load, reference, noise or physical sample time is inferred.
        Split at all physical changes/aperture boundaries. Accepted states use
        the finer partition. Reported partition disagreement checks endpoint,
        all state means and observed endpoint/means, but is not a rigorous error
        certificate. Failed propagation leaves state and boundary charge unapplied.
        A non-committing preview permits a coupled voltage-boundary wrapper to
        validate physical work/storage before accepting the same complete state.
        """
        if np.iscomplexobj(duration_seconds):
            raise ValueError("duration_seconds must be real")
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        shunts = _array("conductance_to_reference_siemens", conductance_to_reference_siemens, (self._size,))
        conductance = passive_matrix(shunts=shunts, edges=resistor_edges, positive_shunts=False)
        if passive_stars is not None:
            if not isinstance(passive_stars, PassiveStarConductance) or passive_stars.shape != conductance.shape:
                raise ValueError("passive_stars must be a validated star network in the same state basis")
            conductance = _ConductanceWithStars(conductance, passive_stars)
        if not isinstance(commit, bool):
            raise ValueError("commit must be boolean")
        current = _array("drive_current_amp", drive_current_amp, (self._size,))
        injected = _array("injected_charge_coulomb", injected_charge_coulomb, (self._size,))
        if np.iscomplexobj(observation_matrix):
            raise ValueError("observation_matrix must be real")
        observation = csr_matrix(observation_matrix, dtype=float, copy=True)
        if (observation.shape[1] != self._size or observation.shape[0] < 1
                or not np.all(np.isfinite(observation.data))):
            raise ValueError("observation_matrix must have finite shape (outputs, states)")
        feedthrough = _array("feedthrough_volts", feedthrough_volts, (observation.shape[0],))
        start, boundary = self._transition(self._capacitance, self._factor, injected, controls)
        derivative = self._factor.solve(current)
        # A deterministic diagonal approximation is used only for exponential
        # preconditioning, never as a substitute for the actual coupled C.
        # SciPy explicitly permits an approximate traceA (see specification).
        trace_rate = -float(np.sum(conductance.diagonal() / self._capacitance.diagonal()))
        if not np.all(np.isfinite(derivative)) or not math.isfinite(trace_rate):
            raise SolverConvergenceError("drive derivative or trace preconditioner is not finite")
        counter = _WorkCounter(controls.max_operator_products)

        def propagate(parts):
            voltage = start.copy()
            mean = np.zeros(self._size)
            step = duration / parts
            operator = _IntervalOperator(self._factor, conductance, derivative, step, counter)
            for _ in range(parts):
                initial = np.concatenate((voltage, [1.0], np.zeros(self._size)))
                result = expm_multiply(operator, initial, traceA=trace_rate * step)
                if not np.all(np.isfinite(result)):
                    raise SolverConvergenceError("nonfinite matrix-exponential result")
                voltage = result[:self._size]
                mean += result[self._size + 1:] / parts
            return voltage, mean

        def scaled_error(left, right):
            denominator = controls.voltage_atol + controls.relative_tolerance * np.maximum(np.abs(left), np.abs(right))
            return float(np.max(np.abs(left - right) / denominator))

        coarse_end, coarse_mean = propagate(1)
        for level in range(1, controls.max_refinements + 1):
            parts = 2**level
            end, mean = propagate(parts)
            output_end, output_mean = observation @ end + feedthrough, observation @ mean + feedthrough
            error = max(scaled_error(end, coarse_end), scaled_error(mean, coarse_mean),
                        scaled_error(output_end, observation @ coarse_end + feedthrough),
                        scaled_error(output_mean, observation @ coarse_mean + feedthrough))
            if math.isfinite(error) and error <= 1:
                start_charge, start_energy = _charge_energy(self._capacitance, start)
                end_charge, end_energy = _charge_energy(self._capacitance, end)
                work = float(duration * (current @ mean))
                if not all(np.all(np.isfinite(value)) for value in (output_end, output_mean, work)):
                    raise SolverConvergenceError("nonfinite output or drive work")
                interval = NetworkInterval(_tuple(start), _tuple(end), _tuple(mean),
                                           _tuple(output_end), _tuple(output_mean), _tuple(start_charge),
                                           _tuple(end_charge), start_energy, end_energy, work, duration)
                result = SparseInterval(boundary, interval, float(np.max(np.abs(end - coarse_end))),
                                        float(np.max(np.abs(mean - coarse_mean))), error, parts, counter.products)
                if commit:
                    self._volts = end.copy()
                    self._pending_preview = None
                else:
                    self._pending_preview = result
                return result
            coarse_end, coarse_mean = end, mean
        raise SolverConvergenceError("partition agreement exceeds declared voltage tolerance")

    def commit_preview(self, result):
        """Accept only this instance's latest unchanged non-committed interval."""
        if result is not self._pending_preview or result is None:
            raise ValueError("result is not the current pending preview")
        if self.volts != result.boundary.transition.before_volts:
            raise ValueError("state changed since preview")
        self._volts = np.array(result.observation.end_volts)
        self._pending_preview = None


class NeurocamPassiveArray:
    """Fixed 64x64 pad nodes plus 64 shared-source nodes, one TFT path/pad.

    A legal row activates that row's caller-specified finite on conductances;
    every other pad retains its explicit finite off path. Observation banks
    select source readout only. Exact bank legality, actual ADC conversions,
    clocks, CPE/noise/clipping and complete resource accounting are external.
    The explicit drive conductance is only a supplied resistive input path,
    not an inferred or complete passive contact model.
    """
    ROWS = 64
    SOURCES = 64
    PADS = 4096
    STATES = 4160

    def __init__(self, *, initial_pad_volts, initial_source_volts,
                 pad_capacitance_farad, source_capacitance_farad, mutual_capacitance_edges,
                 tft_on_conductance_siemens, tft_off_conductance_siemens,
                 pad_drive_conductance_siemens, source_load_conductance_siemens,
                 other_resistor_edges):
        shape = (self.ROWS, self.SOURCES)
        self._on = _array("tft_on_conductance_siemens", tft_on_conductance_siemens, shape).ravel()
        self._off = _array("tft_off_conductance_siemens", tft_off_conductance_siemens, shape).ravel()
        if np.any(self._on <= 0) or np.any(self._off <= 0):
            raise ValueError("every pad needs explicit finite positive on/off conductance")
        self._drive = _array("pad_drive_conductance_siemens", pad_drive_conductance_siemens, shape).ravel()
        load = _array("source_load_conductance_siemens", source_load_conductance_siemens, (self.SOURCES,))
        if np.any(self._drive < 0) or np.any(load < 0):
            raise ValueError("drive/load conductances must be nonnegative")
        self._shunts = np.concatenate((self._drive, load))
        self._other_edges = _edges("other_resistor_edges", other_resistor_edges, self.STATES)
        self._pads = np.arange(self.PADS)
        self._sources = self.PADS + self._pads % self.SOURCES
        initial = np.concatenate((_array("initial_pad_volts", initial_pad_volts, shape).ravel(),
                                  _array("initial_source_volts", initial_source_volts, (self.SOURCES,))))
        capacitance = np.concatenate((_array("pad_capacitance_farad", pad_capacitance_farad, shape).ravel(),
                                      _array("source_capacitance_farad", source_capacitance_farad, (self.SOURCES,))))
        names = tuple(f"pad_{row:02d}_{column:02d}" for row in range(64) for column in range(64))
        names += tuple(f"source_{column:02d}" for column in range(64))
        self.network = SparsePassiveRCNetwork(state_ids=names, initial_volts=initial,
                                             capacitance_to_reference_farad=capacitance,
                                             mutual_capacitance_edges=mutual_capacitance_edges)

    def advance_row(self, *, gate_row, duration_seconds, pad_drive_volts, observed_sources,
                    injected_charge_coulomb, controls):
        if isinstance(gate_row, bool) or not isinstance(gate_row, int) or not 0 <= gate_row < self.ROWS:
            raise ValueError("gate_row must select one of the 64 physical rows")
        selected = tuple(observed_sources)
        if (not selected or any(isinstance(value, bool) or not isinstance(value, int)
                                or not 0 <= value < self.SOURCES for value in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("observed_sources must name distinct physical source lines")
        conductance = np.where(self._pads // self.SOURCES == gate_row, self._on, self._off)
        tft_edges = np.column_stack((self._pads, self._sources, conductance))
        resistor_edges = np.vstack((tft_edges, self._other_edges))
        drive_volts = _array("pad_drive_volts", pad_drive_volts, (self.ROWS, self.SOURCES)).ravel()
        current = np.concatenate((self._drive * drive_volts, np.zeros(self.SOURCES)))
        observation = coo_matrix((np.ones(len(selected)), (np.arange(len(selected)), self.PADS + np.array(selected))),
                                 shape=(len(selected), self.STATES)).tocsr()
        return self.network.advance(duration_seconds=duration_seconds,
                                    conductance_to_reference_siemens=self._shunts,
                                    resistor_edges=resistor_edges, drive_current_amp=current,
                                    observation_matrix=observation, feedthrough_volts=np.zeros(len(selected)),
                                    injected_charge_coulomb=injected_charge_coulomb, controls=controls)
