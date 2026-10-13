"""Small passive R2 circuit kernel from electrical_model_and_probes.md, sec. 4.

C dx/dt = -G x + B v; y = L x + D v. All matrices, inputs, initial state
and intervals are explicit. This dense implementation supports at most 64
electrical states; it is not a 4096-pad array implementation or a calibrated
NeuroCam model. The limit is computational, not a change to the study lattice.

The fixed named state basis must retain off-gate/unobserved circuit nodes.
Changing an observation or a scan interval never resets those states. A change
of capacitance conserves nodal charge, with caller-specified injected charge.
Contact CPE realization, noise/cross-spectra, nonlinear clipping, actual DAQ
clocks and evidence-supported circuit parameters remain separate components.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import expm


MAX_DENSE_STATES = 64


def _real_array(name, values, shape=None):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    array = np.array(values, dtype=float, copy=True)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _symmetric(name, values, size):
    matrix = _real_array(name, values, (size, size))
    # Do not silently symmetrize a caller's physical matrix.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric in the fixed state basis")
    return matrix


def _capacitance(values, size):
    matrix = _symmetric("capacitance_farad", values, size)
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("capacitance_farad must be positive definite") from error
    return matrix


def _conductance(values, size):
    matrix = _symmetric("conductance_siemens", values, size)
    # Reject even a small computed negative eigenvalue: a small C can turn
    # a negative G mode into rapid active growth. This deliberately refuses
    # some mathematically semidefinite matrices with negative roundoff;
    # there is no projection, clipping or approximate-passivity allowance.
    if np.any(np.diag(matrix) < 0) or np.linalg.eigvalsh(matrix)[0] < 0:
        raise ValueError("conductance_siemens must be positive semidefinite")
    return matrix


def _tuple(values):
    return tuple(float(value) for value in values)


def _charge_and_energy(capacitance, volts):
    with np.errstate(over="raise", invalid="raise"):
        try:
            charge = capacitance @ volts
            energy = float(0.5 * (volts @ charge))
        except FloatingPointError as error:
            raise ValueError("charge or stored energy is not representable") from error
    if not np.all(np.isfinite(charge)) or not math.isfinite(energy):
        raise ValueError("charge or stored energy must be finite")
    return charge, energy


@dataclass(frozen=True)
class ChargeTransition:
    before_volts: tuple
    after_volts: tuple
    before_charge_coulomb: tuple
    injected_charge_coulomb: tuple
    after_charge_coulomb: tuple
    before_energy_joule: float
    after_energy_joule: float
    stored_energy_change_joule: float


@dataclass(frozen=True)
class NetworkInterval:
    start_volts: tuple
    end_volts: tuple
    mean_volts: tuple
    output_end_volts: tuple
    output_mean_volts: tuple
    start_charge_coulomb: tuple
    end_charge_coulomb: tuple
    start_energy_joule: float
    end_energy_joule: float
    input_work_joule: float
    duration_seconds: float


class PassiveRCNetwork:
    """Persistent small linear network in one fixed named nodal basis.

    C must be real symmetric positive definite; G must be real symmetric
    positive semidefinite, with no computed negative eigenvalue allowed.
    A mathematically semidefinite G can be refused due to negative roundoff;
    it is never projected or clipped to make it admissible.
    Singular C, ideal algebraic constraints and active transistor dynamics
    need a different solver; this class does not approximate them implicitly.
    Arrays are copied and returned state snapshots are immutable tuples.
    """

    def __init__(self, *, state_ids, initial_volts, capacitance_farad):
        if isinstance(state_ids, (str, bytes)):
            raise ValueError("state_ids must be a sequence of distinct names")
        ids = tuple(state_ids)
        if not 1 <= len(ids) <= MAX_DENSE_STATES:
            raise ValueError(f"this dense kernel supports 1..{MAX_DENSE_STATES} states")
        if any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("state_ids must contain distinct nonempty names")
        volts = _real_array("initial_volts", initial_volts, (len(ids),))
        capacitance = _capacitance(capacitance_farad, len(ids))
        _charge_and_energy(capacitance, volts)
        self._state_ids = ids
        self._volts = volts
        self._capacitance = capacitance

    @property
    def state_ids(self):
        return self._state_ids

    @property
    def volts(self):
        return _tuple(self._volts)

    def switch_capacitance(self, *, capacitance_farad, injected_charge_coulomb):
        """Apply C_new x_after = C_old x_before + q_injected atomically.

        The state basis and its order cannot change. Passing the same C gives
        the fixed-capacitance charge jump. Explicit zero injected charge means
        conserved nodal charge, not unchanged voltage or unchanged energy.
        Reported energy change is stored electrostatic energy only: controller
        work, switching losses and power cannot be inferred from it alone.
        """
        size = len(self._state_ids)
        capacitance = _capacitance(capacitance_farad, size)
        injected = _real_array("injected_charge_coulomb", injected_charge_coulomb, (size,))
        charge_before, energy_before = _charge_and_energy(self._capacitance, self._volts)
        with np.errstate(over="raise", invalid="raise"):
            try:
                after = np.linalg.solve(capacitance, charge_before + injected)
                charge_after, energy_after = _charge_and_energy(capacitance, after)
                energy_change = energy_after - energy_before
            except FloatingPointError as error:
                raise ValueError("charge transition is not representable") from error
        if not np.all(np.isfinite(after)) or not math.isfinite(energy_change):
            raise ValueError("charge transition must be finite")
        result = ChargeTransition(
            self.volts, _tuple(after), _tuple(charge_before), _tuple(injected),
            _tuple(charge_after), energy_before, energy_after, energy_change,
        )
        self._capacitance = capacitance
        self._volts = after
        return result

    def advance(
        self, *, duration_seconds, conductance_siemens, input_conductance_siemens,
        observation_matrix, feedthrough_matrix, input_volts,
    ):
        """Propagate one constant-parameter, constant-input interval exactly.

        B = input_conductance_siemens, L = observation_matrix, and
        D = feedthrough_matrix. B maps input volts to nodal drive current;
        L/D define an observation, not a removal or grounding of unread nodes.
        Supply all arrays explicitly, including zero drive/feedthrough.

        Split at every input, gate, circuit-parameter or aperture boundary and
        advance discarded settling intervals too. The interval mean permits a
        configured aperture integral; it does not choose the physical ADC's
        aperture. No sample, conversion, settling or switching resource count
        is inferred here. Input work is integral(x.T B v dt), only the drive
        work into these modeled states, not complete hardware energy.
        """
        duration = float(duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        size = len(self._state_ids)
        conductance = _conductance(conductance_siemens, size)
        inputs = _real_array("input_volts", input_volts)
        if inputs.ndim != 1 or inputs.size < 1:
            raise ValueError("input_volts must be a nonempty vector")
        drive = _real_array("input_conductance_siemens", input_conductance_siemens, (size, inputs.size))
        observation = _real_array("observation_matrix", observation_matrix)
        if observation.ndim != 2 or observation.shape[1] != size or observation.shape[0] < 1:
            raise ValueError("observation_matrix must have shape (outputs, states)")
        feedthrough = _real_array("feedthrough_matrix", feedthrough_matrix, (observation.shape[0], inputs.size))
        charge_start, energy_start = _charge_and_energy(self._capacitance, self._volts)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                current = drive @ inputs
                # Normalize time to [0, 1]. The last block then accumulates
                # mean(x), avoiding division by a very short physical dt.
                generator = np.zeros((2 * size + 1, 2 * size + 1))
                generator[:size, :size] = -np.linalg.solve(self._capacitance, conductance) * duration
                generator[:size, size] = np.linalg.solve(self._capacitance, current) * duration
                generator[size + 1:, :size] = np.eye(size)
                if not np.all(np.isfinite(generator)):
                    raise ValueError("interval generator must be finite")
                initial = np.concatenate((self._volts, [1.0], np.zeros(size)))
                propagated = expm(generator) @ initial
                end = propagated[:size]
                mean = propagated[size + 1:]
                output_end = observation @ end + feedthrough @ inputs
                output_mean = observation @ mean + feedthrough @ inputs
                charge_end, energy_end = _charge_and_energy(self._capacitance, end)
                input_work = float(duration * (current @ mean))
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("interval propagation is not representable") from error
        if not all(np.all(np.isfinite(value)) for value in (propagated, output_end, output_mean, input_work)):
            raise ValueError("interval propagation must be finite")
        result = NetworkInterval(
            self.volts, _tuple(end), _tuple(mean), _tuple(output_end), _tuple(output_mean),
            _tuple(charge_start), _tuple(charge_end), energy_start, energy_end,
            input_work, duration,
        )
        self._volts = end.copy()
        return result
