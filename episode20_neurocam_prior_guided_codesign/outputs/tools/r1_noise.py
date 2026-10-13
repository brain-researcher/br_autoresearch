"""Explicit held current ports for a caller-owned one-state R1 reduction.

The caller supplies the conjugate reduction ``I_eq = sum(b_i * i_i)`` and
physical noise-port voltages ``y_i = b_i * x + d_i``. The same coefficient
must appear in both expressions. Original currents and voltage offsets are
kept, including ports whose net state injection cancels or whose b_i is zero.
This module neither chooses that physical reduction nor derives a temperature,
PSD, contact model, latent draw or a complete acquisition architecture.

For one held interval, W_i = duration * i_i * (b_i * mean(x) + d_i).
The reported offset contribution is algebraic noise-port work. It is NOT an
automatic restoration of the work of eliminated ideal voltage sources, nor
a claim about total device energy. Those require the caller's physical model.
"""

from dataclasses import dataclass
import math
from numbers import Real


def _name(label, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty name")
    return value


def _real(label, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be real")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _bounded_tuple(label, values, maximum):
    result = []
    for value in values:
        if len(result) >= maximum:
            raise ValueError(f"{label} exceeds its finite bound")
        result.append(value)
    return tuple(result)


def _vector(label, values, count):
    result = _bounded_tuple(label, values, count)
    if len(result) != count:
        raise ValueError(f"{label} must contain exactly {count} entries")
    return tuple(_real(label, value) for value in result)


def _sum(label, values):
    try:
        return _real(label, math.fsum(values))
    except OverflowError as error:
        raise ValueError(f"{label} is not finite") from error


@dataclass(frozen=True)
class R1NoiseLimits:
    max_ports: int
    max_stored_numeric_bytes: int

    def __post_init__(self):
        for name in ("max_ports", "max_stored_numeric_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class R1CurrentPort:
    port_id: str
    state_current_coefficient: float
    contribution_role: str
    owned_component_id: str

    def __post_init__(self):
        for name in ("port_id", "contribution_role", "owned_component_id"):
            _name(name, getattr(self, name))
        object.__setattr__(self, "state_current_coefficient",
                           _real("state_current_coefficient", self.state_current_coefficient))


@dataclass(frozen=True)
class R1NoiseReduction:
    """Named, immutable reduction and a cap for its one-live-sample payload.

    Stored numeric payload is 3 * P * 8 bytes: P coefficients plus P original
    currents and P voltage offsets in one sample. Python/string overhead,
    transient arithmetic and returned work records are excluded. An acquisition
    adapter must separately bound its live samples and emitted records.
    """

    state_id: str
    voltage_reference_id: str
    reduction_reference: str
    ports: tuple[R1CurrentPort, ...]
    limits: R1NoiseLimits

    def __post_init__(self):
        for name in ("state_id", "voltage_reference_id", "reduction_reference"):
            _name(name, getattr(self, name))
        if not isinstance(self.limits, R1NoiseLimits):
            raise ValueError("explicit R1NoiseLimits required")
        maximum = min(self.limits.max_ports, self.limits.max_stored_numeric_bytes // 24)
        ports = _bounded_tuple("ports", self.ports, maximum)
        if not ports or any(not isinstance(port, R1CurrentPort) for port in ports):
            raise ValueError("at least one typed current port is required")
        if len({port.port_id for port in ports}) != len(ports):
            raise ValueError("duplicate port_id")
        if len({port.owned_component_id for port in ports}) != len(ports):
            raise ValueError("a physical component cannot be owned twice")
        object.__setattr__(self, "ports", ports)

    @property
    def retained_numeric_bytes(self):
        return 24 * len(self.ports)

    def sample_from_values(self, current_amplitudes_amp, *, port_voltage_offsets_volts):
        return R1HeldCurrentNoise(self, current_amplitudes_amp, port_voltage_offsets_volts)


@dataclass(frozen=True)
class R1NoiseWork:
    port_work_joule: tuple[tuple[str, float], ...]
    noise_current_source_interval_work_joule: float
    equivalent_nodal_noise_interval_work_joule: float
    known_port_voltage_offset_work_joule: float


@dataclass(frozen=True)
class R1HeldCurrentNoise:
    """One owned draw, held unchanged through solver/aperture/storage splits."""

    reduction: R1NoiseReduction
    current_amplitudes_amp: tuple[float, ...]
    port_voltage_offsets_volts: tuple[float, ...]

    def __post_init__(self):
        if not isinstance(self.reduction, R1NoiseReduction):
            raise ValueError("typed caller-owned R1NoiseReduction required")
        count = len(self.reduction.ports)
        object.__setattr__(self, "current_amplitudes_amp",
                           _vector("current amplitudes", self.current_amplitudes_amp, count))
        object.__setattr__(self, "port_voltage_offsets_volts",
                           _vector("port voltage offsets", self.port_voltage_offsets_volts, count))
        # Refuse a nonfinite projection before any state transition.
        self.equivalent_current_amp

    @property
    def numeric_values(self):
        """Original amplitudes followed by offsets, for a future replay digest."""
        return self.current_amplitudes_amp + self.port_voltage_offsets_volts

    @property
    def equivalent_current_amp(self):
        return _sum("equivalent current", (
            _real("projected port current", port.state_current_coefficient * current)
            for port, current in zip(self.reduction.ports, self.current_amplitudes_amp)
        ))

    def work(self, *, mean_state_volts, duration_seconds):
        mean = _real("mean_state_volts", mean_state_volts)
        duration = _real("duration_seconds", duration_seconds)
        if duration <= 0:
            raise ValueError("duration_seconds must be positive")
        per_port = []
        offsets = []
        for port, current, offset in zip(self.reduction.ports,
                                         self.current_amplitudes_amp,
                                         self.port_voltage_offsets_volts):
            voltage = _real("physical noise-port voltage",
                            port.state_current_coefficient * mean + offset)
            power = _real("physical noise-port power", current * voltage)
            per_port.append((port.port_id, _real("physical noise-port work", duration * power)))
            offset_power = _real("port offset power", current * offset)
            offsets.append(_real("port offset work", duration * offset_power))
        nodal_power = _real("equivalent nodal noise power", self.equivalent_current_amp * mean)
        return R1NoiseWork(
            tuple(per_port),
            _sum("total physical noise-port work", (value for _, value in per_port)),
            _real("equivalent nodal noise work", duration * nodal_power),
            _sum("known port voltage-offset work", offsets),
        )
