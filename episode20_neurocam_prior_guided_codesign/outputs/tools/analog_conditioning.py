"""Caller-supplied conditional R3 buffer/filter macromodel, without defaults.

For each of the 64 existing source lines, append buffer ``b`` and filter ``f``
states. With ``u = clip(gain * v_source + offset, lower, upper)``:

    C_b b' = g_b (u - b) - g_out (b - f)
    C_f f' = g_out (b - f) - g_leak f - physical mux currents.

Input G/C load the original source. The joint circuit solver owns every state,
aperture integral and boundary charge; this module never advances a separate
filter from source interval means. The rails limit the internal voltage drive,
not the loaded output node. Backdrive can exceed them. Finite bandwidth gives
persistent overload recovery, without a hard output clamp, slew-rate law or
extra overload-storage mechanism. This is an explicit conditional model, not a
calibrated NeuroCam amplifier or an identification of NI circuitry.

Controlled-source terminal work is distinct from nodal Norton work and from
actual supply consumption. No quiescent current, supply efficiency, noise,
physical parameter values, ADC conversions or extra ADC groups are inferred.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be an explicit finite real number")
    return float(value)


@dataclass(frozen=True)
class SourceConditioning:
    gain: float
    offset_volts: float
    drive_lower_volts: float
    drive_upper_volts: float
    input_capacitance_farad: float
    input_conductance_siemens: float
    buffer_capacitance_farad: float
    buffer_conductance_siemens: float
    output_conductance_siemens: float
    filter_capacitance_farad: float
    filter_leak_conductance_siemens: float
    initial_buffer_volts: float
    initial_filter_volts: float

    def __post_init__(self):
        for name, value in vars(self).items():
            object.__setattr__(self, name, _number(value, name))
        if self.drive_lower_volts >= self.drive_upper_volts:
            raise ValueError("internal drive rails must be strictly ordered")
        for name in ("buffer_capacitance_farad", "buffer_conductance_siemens",
                     "output_conductance_siemens", "filter_capacitance_farad"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("input_capacitance_farad", "input_conductance_siemens",
                     "filter_leak_conductance_siemens"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class ControlledDriveEvaluation:
    node_indices: tuple
    current_amp: np.ndarray
    controlled_voltage_source_power_watt: float
    controlled_branch_dissipation_watt: float
    controlled_nodal_current_power_watt: float
    buffer_shunt_power_watt: float


@dataclass(frozen=True, init=False, eq=False)
class ControlledVoltageDrives:
    """Owned 64-source nonlinear injection and sparse Jacobian action.

    Positive ``g_b`` damping belongs to the separately stamped passive shunt.
    This object supplies only the dependent ``g_b*u`` injection. The Jacobian
    uses zero slope exactly at either clip kink (a semismooth convention), not
    a smoothed or shifted rail. Numeric outputs are new arrays, never aliases of
    retained configuration. Private numeric storage is counted separately from
    the immutable Python configuration/stamp objects.
    """
    def __init__(self, *, channels, source_nodes, buffer_nodes, state_count):
        channels = tuple(channels)
        source_nodes, buffer_nodes = tuple(source_nodes), tuple(buffer_nodes)
        if (type(state_count) is not int or state_count < 1
                or len(channels) != 64
                or any(not isinstance(channel, SourceConditioning) for channel in channels)
                or len(source_nodes) != 64 or len(buffer_nodes) != 64):
            raise ValueError("controlled drives require all 64 declared source/buffer pairs")
        nodes = source_nodes + buffer_nodes
        if (any(type(node) is not int or not 0 <= node < state_count for node in nodes)
                or len(set(nodes)) != len(nodes)):
            raise ValueError("source and buffer nodes must be distinct and in range")
        for name, value in (("_state_count", state_count), ("_source_nodes", source_nodes),
                            ("_buffer_nodes", buffer_nodes),
                            ("_source_index", np.array(source_nodes, dtype=np.int64)),
                            ("_buffer_index", np.array(buffer_nodes, dtype=np.int64)),
                            ("_gain", np.array([channel.gain for channel in channels])),
                            ("_offset", np.array([channel.offset_volts for channel in channels])),
                            ("_lower", np.array([channel.drive_lower_volts for channel in channels])),
                            ("_upper", np.array([channel.drive_upper_volts for channel in channels])),
                            ("_conductance", np.array([channel.buffer_conductance_siemens for channel in channels]))):
            object.__setattr__(self, name, value)
        for value in self._numeric_arrays:
            value.flags.writeable = False

    @property
    def _numeric_arrays(self):
        return (self._source_index, self._buffer_index, self._gain, self._offset,
                self._lower, self._upper, self._conductance)

    @property
    def state_count(self):
        return self._state_count

    @property
    def source_nodes(self):
        return self._source_nodes

    @property
    def buffer_nodes(self):
        return self._buffer_nodes

    @property
    def buffer_conductance_siemens(self):
        """Required passive damping in buffer_nodes order; extra loads may add to it."""
        return tuple(float(value) for value in self._conductance)

    @property
    def numeric_storage_bytes(self):
        return sum(value.nbytes for value in self._numeric_arrays)

    def with_input_noise_volts(self, values):
        """Return one held input-noise view, leaving the base drive unchanged.

        Input-referred voltage enters before gain and rail clipping. Its effect
        on controlled-source terminal work is already in evaluate(); this is
        not a separately physical input power supply or an added output noise.
        """
        if np.iscomplexobj(values):
            raise ValueError("amplifier input noise must be real")
        values = np.asarray(values, dtype=float)
        if values.shape != (64,) or not np.all(np.isfinite(values)):
            raise ValueError("amplifier input noise must cover the 64 original source lines")
        with np.errstate(over="raise", invalid="raise"):
            try:
                offset = self._offset + self._gain * values
            except FloatingPointError as error:
                raise ValueError("noise-shifted controlled voltage must be representable") from error
        if not np.all(np.isfinite(offset)):
            raise ValueError("noise-shifted controlled voltage must be finite")
        offset.flags.writeable = False
        result = object.__new__(ControlledVoltageDrives)
        for name, value in vars(self).items():
            object.__setattr__(result, name, offset if name == "_offset" else value)
        return result

    def _vector(self, values, name):
        if np.iscomplexobj(values):
            raise ValueError(f"{name} must be real")
        values = np.asarray(values, dtype=float)
        if values.shape != (self._state_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite in the complete state basis")
        return values

    def _target(self, volts):
        with np.errstate(over="raise", invalid="raise"):
            try:
                requested = self._gain * volts[self._source_index] + self._offset
            except FloatingPointError as error:
                raise ValueError("controlled voltage request must be representable") from error
        return requested, np.clip(requested, self._lower, self._upper)

    def evaluate(self, volts):
        volts = self._vector(volts, "volts")
        _, target = self._target(volts)
        buffer = volts[self._buffer_index]
        with np.errstate(over="raise", invalid="raise"):
            try:
                current = self._conductance * target
                branch_current = self._conductance * (target - buffer)
                source_power = float(np.sum(target * branch_current))
                dissipation = float(np.sum((target - buffer) * branch_current))
                nodal_power = float(np.sum(buffer * current))
                shunt_power = float(np.sum(self._conductance * buffer * buffer))
            except FloatingPointError as error:
                raise ValueError("controlled drive current and power must be representable") from error
        if not (np.all(np.isfinite(current)) and all(math.isfinite(value) for value in
                (source_power, dissipation, nodal_power, shunt_power))):
            raise ValueError("controlled drive current and power must be finite")
        return ControlledDriveEvaluation(self._buffer_nodes, current, source_power,
                                         dissipation, nodal_power, shunt_power)

    def jacobian_product(self, volts, direction):
        volts = self._vector(volts, "volts")
        direction = self._vector(direction, "direction")
        requested, _ = self._target(volts)
        inside = (requested > self._lower) & (requested < self._upper)
        result = np.zeros(64)
        with np.errstate(over="raise", invalid="raise"):
            try:
                result[inside] = (self._conductance[inside] * self._gain[inside]
                                  * direction[self._source_index[inside]])
            except FloatingPointError as error:
                raise ValueError("controlled drive Jacobian action must be representable") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("controlled drive Jacobian action must be finite")
        return result


@dataclass(frozen=True)
class ConditioningStamp:
    source_capacitance_farad: tuple
    source_conductance_siemens: tuple
    added_capacitance_farad: tuple
    added_shunts_siemens: tuple
    initial_volts: tuple
    state_ids: tuple
    resistor_edges: tuple
    output_nodes: tuple
    controlled_drives: ControlledVoltageDrives


@dataclass(frozen=True)
class AnalogConditioning:
    channels: tuple
    source_load_role: str
    source_capacitance_role: str
    voltage_reference_id: str

    def __post_init__(self):
        channels = tuple(self.channels)
        if len(channels) != 64 or any(not isinstance(channel, SourceConditioning) for channel in channels):
            raise ValueError("conditioning requires exactly all 64 original source lines")
        if self.source_load_role != "additional_fixed_shunt_excludes_conditioning":
            raise ValueError("existing source shunts must explicitly exclude conditioning input loads")
        if self.source_capacitance_role != "additional_fixed_capacitance_excludes_conditioning":
            raise ValueError("existing source capacitance must explicitly exclude conditioning input capacitance")
        if not isinstance(self.voltage_reference_id, str) or not self.voltage_reference_id.strip():
            raise ValueError("conditioning requires an explicit common voltage reference")
        object.__setattr__(self, "channels", channels)

    @property
    def state_count(self):
        return 128

    def stamp(self, *, state_count, source_node_offset, first_conditioning_state,
              voltage_reference_id):
        """Bind once and append all 64 buffers followed by all 64 filters.

        The assembler adds input G/C exactly once, then routes each existing
        mux source membership to ``output_nodes[source_index]``. Original source
        diagnostics remain available but are not conditioned ADC inputs.
        All returned coefficients and initial values are immutable tuples.
        """
        if voltage_reference_id != self.voltage_reference_id:
            raise ValueError("conditioning voltage reference differs from the array")
        if (any(type(value) is not int for value in
                (state_count, source_node_offset, first_conditioning_state))
                or source_node_offset < 0 or first_conditioning_state < source_node_offset + 64
                or first_conditioning_state + self.state_count > state_count):
            raise ValueError("conditioning state placement must follow the complete original source axis")
        sources = tuple(range(source_node_offset, source_node_offset + 64))
        buffers = tuple(first_conditioning_state + source for source in range(64))
        filters = tuple(first_conditioning_state + 64 + source for source in range(64))
        channels = self.channels
        return ConditioningStamp(
            tuple(channel.input_capacitance_farad for channel in channels),
            tuple(channel.input_conductance_siemens for channel in channels),
            tuple(channel.buffer_capacitance_farad for channel in channels)
                + tuple(channel.filter_capacitance_farad for channel in channels),
            tuple(channel.buffer_conductance_siemens for channel in channels)
                + tuple(channel.filter_leak_conductance_siemens for channel in channels),
            tuple(channel.initial_buffer_volts for channel in channels)
                + tuple(channel.initial_filter_volts for channel in channels),
            tuple(f"conditioning_buffer_{source:02d}" for source in range(64))
                + tuple(f"conditioning_filter_{source:02d}" for source in range(64)),
            tuple((buffer, filtered, channel.output_conductance_siemens)
                  for buffer, filtered, channel in zip(buffers, filters, channels)),
            filters,
            ControlledVoltageDrives(channels=channels, source_nodes=sources,
                                    buffer_nodes=buffers, state_count=state_count))
