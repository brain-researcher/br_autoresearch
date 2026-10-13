"""Physical noise placement and eliminated-contact work, without a PSD recipe.

All amplitudes are held by the caller over one physical input interval. They
are the original current/voltage-source coordinates, never independently drawn
post-elimination nodal currents. For a contact star eta points junction->node,
and access voltage e adds to field port U. With d=1+R*sum(h):

  a = (U+e+R*h.v-R*sum(eta))/d
  retained noise = eta - R*h*sum(eta)/d + h*e/d.

This preserves the induced correlations and the physical source work. No PSD,
temperature, Gaussianity, independence, bandwidth, thermal source for a biased
device, or noise absence is inferred. A zero-access-resistance branch requires
zero access-noise voltage. A zero star conductance owns no resistor-noise port
amplitude. Other conditional current sources must be named separately.

Star ownership is not all contact-noise ownership: an ideal interface with
positive access R has a separately retained charge-transfer resistor. The array
factory must require its explicit current port or recorded omission. Retained
noise-current work is signed; input-referred amplifier voltage noise changes
the existing controlled drive before gain/clip and has no invented supply-work
term. Python object overhead and transient arrays are outside numeric limits.

Optional external-DAQ branch current ports are appended after existing noise
coordinates. Their actual track/off path and external U must be supplied at
interval validation: eta enters the hold, its work is eta*(mean_hold-U)*dt,
and the external ideal source receives the separate U*eta*dt correction.
Zero-conductance branches require explicit zero amplitudes. The full assembled
branch inventory owns those component IDs; a nodal source cannot also own one.
"""

from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from daq_mux_hold import (ExternalDAQBranch, ExternalDAQBranchDrive,
                          external_daq_branch_id)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be an explicit nonempty name")
    return value


def _bounded_tuple(values, limit, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} requires a finite sequence, not text")
    result = []
    for value in values:
        if len(result) >= limit:
            raise ValueError(f"{name} exceeds its finite item limit")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class NoiseLimits:
    max_sources: int
    max_stored_numeric_bytes: int

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in vars(self).values()):
            raise ValueError("noise source/numeric limits must be positive integers")


@dataclass(frozen=True)
class NodalCurrentPort:
    port_id: str
    positive_state_id: str
    negative_state_id: str | None
    contribution_role: str
    owned_component_id: str

    def __post_init__(self):
        for field in ("port_id", "positive_state_id", "contribution_role", "owned_component_id"):
            _name(getattr(self, field), field)
        if self.negative_state_id is not None:
            _name(self.negative_state_id, "negative_state_id")
            if self.negative_state_id == self.positive_state_id:
                raise ValueError("a current port needs distinct physical terminals")


@dataclass(frozen=True)
class ExternalDAQCurrentPort:
    """Current eta from the named external ideal source U into its ADC hold.

    The exact assembled branch inventory binds the identity and endpoint.
    This is its switch-path current noise, not a current source from ground.
    Spectrum/temperature and nonzero amplitudes are never inferred from G.
    """
    port_id: str
    branch_id: str
    adc_id: str
    input_id: str
    hold_state_id: str
    contribution_role: str

    def __post_init__(self):
        for field in ("port_id", "hold_state_id", "contribution_role"):
            _name(getattr(self, field), field)
        if self.branch_id != external_daq_branch_id(self.adc_id, self.input_id):
            raise ValueError("external current noise must own its canonical assembled branch identity")

    @property
    def owned_component_id(self):
        return self.branch_id


@dataclass(frozen=True)
class NoiseWork:
    noise_current_source_interval_work_joule: float
    noise_access_voltage_source_interval_work_joule: float
    field_source_interval_work_correction_joule: float
    daq_external_source_interval_work_correction_joule: float = 0.0


@dataclass(frozen=True, init=False, eq=False)
class NoiseSample:
    """One immutable float64 vector; tuple inspection is materialized on request.

    Spectral evaluation supplies a NumPy vector, avoiding a Python type-check
    loop over every physical port at each acquisition event. Owned bytes make
    the retained vector immutable even if the caller later mutates its input.
    """
    layout: object
    _numeric_bytes: bytes = field(repr=False)

    def __init__(self, layout, numeric_values):
        if not isinstance(layout, PhysicalNoiseLayout):
            raise ValueError("a bound physical-noise layout is required")
        if not hasattr(numeric_values, "__len__") or len(numeric_values) != layout.numeric_value_count:
            raise ValueError("noise amplitudes must cover every original physical port")
        if isinstance(numeric_values, np.ndarray):
            if numeric_values.shape != (layout.numeric_value_count,) or numeric_values.dtype.kind not in "fiu":
                raise ValueError("noise amplitudes must be a real numeric vector, excluding booleans")
            values = numeric_values
        else:
            # Preserve strict admission for heterogeneous caller sequences.
            values = tuple(_number(value, "noise amplitude") for value in numeric_values)
        with np.errstate(over="raise", invalid="raise"):
            try:
                packed = np.asarray(values, dtype="<f8").tobytes(order="C")
            except (FloatingPointError, OverflowError) as error:
                raise ValueError("noise amplitudes must be representable finite float64") from error
        frozen = np.frombuffer(packed, dtype="<f8")
        if not np.all(np.isfinite(frozen)):
            raise ValueError("noise amplitudes must be finite")
        layout._validate_values(frozen)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_numeric_bytes", packed)

    @property
    def numeric_values(self):
        return tuple(self.numeric_array)

    @property
    def numeric_array(self):
        """Read-only view of owned bytes in the original physical port order."""
        return np.frombuffer(self._numeric_bytes, dtype="<f8")

    @property
    def state_ids(self):
        return self.layout.state_ids

    @property
    def voltage_reference_id(self):
        return self.layout.voltage_reference_id

    @property
    def sample_numeric_bytes(self):
        return self.layout.sample_numeric_bytes


@dataclass(frozen=True, init=False, eq=False)
class PhysicalNoiseLayout:
    """One owned, immutable full-array port order and linear-time star map.

    The assembler supplies its actual 4,096 contact stars. Contribution roles
    and omissions describe a selected model; they are not evidence of measured
    absence or of a validated thermal spectrum. Limits include original scalar
    topology payloads, cached NumPy coefficients and one live numeric sample.
    """
    def __init__(self, *, state_ids, voltage_reference_id, nodal_current_ports,
                 contact_stars, contact_ownership, contact_noise_role,
                 amplifier_input_noise_role, excluded_retained_contact_components,
                 limits, external_daq_branches=(), external_daq_current_ports=()):
        if not isinstance(limits, NoiseLimits):
            raise ValueError("explicit physical noise limits are required")
        states = _bounded_tuple(state_ids, limits.max_stored_numeric_bytes // 8, "state basis")
        if (not states or len(set(states)) != len(states)
                or any(not isinstance(name, str) or not name for name in states)):
            raise ValueError("noise layout requires the complete distinct state basis")
        _name(voltage_reference_id, "voltage_reference_id")
        if contact_ownership != "all_eliminated_star_branch_and_access_noise_owned_here":
            raise ValueError("explicit eliminated-star noise ownership is required")
        _name(contact_noise_role, "contact_noise_role")
        if amplifier_input_noise_role is not None:
            _name(amplifier_input_noise_role, "amplifier_input_noise_role")
        ports = _bounded_tuple(nodal_current_ports, limits.max_sources, "nodal current ports")
        if (any(not isinstance(port, NodalCurrentPort) for port in ports)
                or len({port.port_id for port in ports}) != len(ports)
                or len({port.owned_component_id for port in ports}) != len(ports)):
            raise ValueError("nodal current sources require unique port and component owners")
        index = {name: i for i, name in enumerate(states)}
        if any(port.positive_state_id not in index
               or port.negative_state_id is not None and port.negative_state_id not in index for port in ports):
            raise ValueError("every current-source terminal must belong to the full state basis")
        external_branches = _bounded_tuple(external_daq_branches, limits.max_sources, "external DAQ branch inventory")
        if (any(not isinstance(branch, ExternalDAQBranch) for branch in external_branches)
                or len({branch.branch_id for branch in external_branches}) != len(external_branches)
                or any(branch.hold_state_id not in index for branch in external_branches)):
            raise ValueError("external inventory needs distinct assembled branches in the full state basis")
        branch_by_id = {branch.branch_id: branch for branch in external_branches}
        external_ports = _bounded_tuple(external_daq_current_ports, limits.max_sources, "external DAQ current ports")
        if (any(not isinstance(port, ExternalDAQCurrentPort) for port in external_ports)
                or len({port.branch_id for port in external_ports}) != len(external_ports)
                or len({port.port_id for port in external_ports}) != len(external_ports)):
            raise ValueError("external current sources require unique physical branch ownership and port IDs")
        for port in external_ports:
            branch = branch_by_id.get(port.branch_id)
            if (branch is None or (port.adc_id, port.input_id, port.hold_state_id)
                    != (branch.adc_id, branch.input_id, branch.hold_state_id)):
                raise ValueError("external current port must match the full assembled branch inventory")
        if any(port.owned_component_id in branch_by_id for port in ports):
            raise ValueError("external DAQ branches cannot also be owned by ground/retained nodal current ports")
        stars = _bounded_tuple(contact_stars, 4096, "contact stars")
        if len(stars) != 4096:
            raise ValueError("exactly 4096 assembled contact stars are required")
        copied, nodes, group, h, resistance, names = [], [], [], [], [], []
        for pad, (indices, conductances, access) in enumerate(stars):
            remaining = (limits.max_sources - len(ports) - len(external_ports) - len(nodes)
                         - 4096 - (64 if amplifier_input_noise_role is not None else 0))
            indices = _bounded_tuple(indices, max(remaining, 0), "contact branch nodes")
            conductances = _bounded_tuple(conductances, len(indices), "contact branch conductances")
            if (not indices or len(indices) != len(conductances) or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < len(states) for i in indices)):
                raise ValueError("contact star nodes must be distinct, retained and in range")
            weights = tuple(_number(value, "star conductance") for value in conductances)
            access = _number(access, "star access resistance")
            if access < 0 or any(value < 0 for value in weights):
                raise ValueError("star conductance/access resistance must be nonnegative")
            copied.append((indices, weights, access))
            nodes.extend(indices)
            group.extend([pad] * len(indices))
            h.extend(weights)
            resistance.append(access)
            names.extend(f"contact:{pad:04d}:star_branch:{branch}" for branch in range(len(indices)))
            if len(ports) + len(external_ports) + len(nodes) + 4096 + (64 if amplifier_input_noise_role is not None else 0) > limits.max_sources:
                raise ValueError("physical noise source-count limit exceeded")
        port_ids = (tuple(port.port_id for port in ports) + tuple(names)
                    + tuple(f"contact:{pad:04d}:star_access_voltage" for pad in range(4096))
                    + (() if amplifier_input_noise_role is None else
                       tuple(f"amplifier_input_voltage:{source:02d}" for source in range(64)))
                    + tuple(port.port_id for port in external_ports))
        if len(set(port_ids)) != len(port_ids):
            raise ValueError("physical noise port identities must not collide")
        automatic_stop = len(port_ids) - len(external_ports)
        automatic_owners = set(port_ids[len(ports):automatic_stop])
        if any(port.owned_component_id in automatic_owners for port in ports):
            raise ValueError("automatically owned contact/amplifier ports cannot also have nodal owners")
        omitted = tuple(_bounded_tuple(item, 2, "contact omission pair") for item in
            _bounded_tuple(excluded_retained_contact_components, 4096, "contact omissions"))
        if (any(len(item) != 2 for item in omitted)
                or len({item[0] for item in omitted}) != len(omitted)):
            raise ValueError("each omitted retained contact component needs one explicit reason")
        for component, reason in omitted:
            _name(component, "omitted component")
            _name(reason, "omission reason")
        count, term_count = len(port_ids), len(nodes)
        # Topology tuple payload: 2*terms+4096; arrays: 5*terms+4*4096+2*ports.
        numeric_bytes = 8 * (7 * term_count + 5 * 4096 + 2 * len(ports) + count
                            + 2 * len(external_branches) + len(external_ports)
                            + sum(value == 0 for value in h) + sum(value == 0 for value in resistance))
        if numeric_bytes > limits.max_stored_numeric_bytes:
            raise ValueError("physical noise retained numeric-byte limit exceeded")
        nodes, group, h = np.array(nodes, dtype=np.int64), np.array(group, dtype=np.int64), np.array(h)
        resistance = np.array(resistance)
        totals = np.bincount(group, weights=h, minlength=4096)
        with np.errstate(over="raise", invalid="raise"):
            try:
                product = resistance * totals
                inverse_d = 1. / (1. + product)
                mix = product * inverse_d
                r_div_d = resistance * inverse_d
                fractions = np.divide(h, totals[group], out=np.zeros_like(h), where=totals[group] > 0)
                p, b = fractions * mix[group], h * inverse_d[group]
            except FloatingPointError as error:
                raise ValueError("contact noise elimination coefficients must be representable") from error
        if not all(np.all(np.isfinite(value)) for value in (totals, inverse_d, mix, r_div_d, fractions, p, b)):
            raise ValueError("contact noise elimination coefficients must be finite")
        positive = np.array([index[port.positive_state_id] for port in ports], dtype=np.int64)
        negative = np.array([-1 if port.negative_state_id is None else index[port.negative_state_id]
                             for port in ports], dtype=np.int64)
        external_nodes = np.array([index[port.hold_state_id] for port in external_ports], dtype=np.int64)
        for key, value in dict(state_ids=states, voltage_reference_id=voltage_reference_id,
                nodal_current_ports=ports, contact_stars=tuple(copied), contact_ownership=contact_ownership,
                contact_noise_role=contact_noise_role, amplifier_input_noise_role=amplifier_input_noise_role,
                excluded_retained_contact_components=omitted, limits=limits, port_ids=port_ids,
                external_daq_branches=external_branches, external_daq_current_ports=external_ports,
                port_units=("A",) * (len(ports) + term_count)
                    + ("V",) * (count - len(ports) - term_count - len(external_ports))
                    + ("A",) * len(external_ports),
                numeric_value_count=count, sample_numeric_bytes=8 * count, retained_numeric_bytes=numeric_bytes,
                _nodes=nodes, _group=group, _totals=totals, _inverse_d=inverse_d, _mix=mix,
                _r_div_d=r_div_d, _fractions=fractions, _p=p, _b=b, _positive=positive, _negative=negative,
                _term_count=term_count, _nodal_count=len(ports),
                _external_nodes=external_nodes, _external_count=len(external_ports),
                _zero_h=tuple(i for i, value in enumerate(h) if value == 0),
                _zero_r=tuple(i for i, value in enumerate(resistance) if value == 0)).items():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
            object.__setattr__(self, key, value)

    def _validate_values(self, values):
        offset = self._nodal_count
        if any(values[offset + i] != 0 for i in self._zero_h):
            raise ValueError("zero-conductance star branches require zero owned branch noise")
        offset += self._term_count
        if any(values[offset + i] != 0 for i in self._zero_r):
            raise ValueError("zero-access-resistance stars require zero owned access voltage noise")

    def sample_from_values(self, values):
        return NoiseSample(self, values)

    def _parts(self, sample):
        if not isinstance(sample, NoiseSample) or sample.layout is not self:
            raise ValueError("noise sample must use this exact bound physical layout")
        values = sample.numeric_array
        first, stop = self._nodal_count, self._nodal_count + self._term_count
        amplifier_end = stop + 4096 + (64 if self.amplifier_input_noise_role is not None else 0)
        return (values[:first], values[first:stop], values[stop:stop + 4096],
                values[stop + 4096:amplifier_end], values[amplifier_end:])

    def _external_context(self, sample, external_branch_drives):
        """Resolve complete actual interval context, including every off path."""
        if not self.external_daq_current_ports:
            return np.empty(0)
        _, _, _, _, current = self._parts(sample)
        supplied = _bounded_tuple(external_branch_drives, len(self.external_daq_branches), "external interval branches")
        if (len(supplied) != len(self.external_daq_branches)
                or any(not isinstance(branch, ExternalDAQBranchDrive) for branch in supplied)
                or len({branch.branch_id for branch in supplied}) != len(supplied)):
            raise ValueError("complete identified external track/off branch context is required")
        actual = {branch.branch_id: branch for branch in supplied}
        shared = {}
        for branch in self.external_daq_branches:
            interval = actual.get(branch.branch_id)
            if (interval is None or interval.hold_state >= len(self.state_ids)
                    or self.state_ids[interval.hold_state] != branch.hold_state_id
                    or interval.voltage_reference_id != self.voltage_reference_id
                    or interval.conductance_siemens not in (branch.track_conductance_siemens,
                                                          branch.off_conductance_siemens)):
                raise ValueError("external interval identity, hold or conductance differs from its inventory")
            if interval.input_id in shared and shared[interval.input_id] != interval.source_volts:
                raise ValueError("shared external source voltages must agree")
            shared[interval.input_id] = interval.source_volts
        voltages = np.empty(self._external_count)
        for i, port in enumerate(self.external_daq_current_ports):
            interval = actual[port.branch_id]
            if interval.conductance_siemens == 0 and current[i] != 0:
                raise ValueError("zero-conductance external DAQ branch requires explicit zero noise current")
            voltages[i] = interval.source_volts
        return voltages

    def validate_external_branch_sample(self, sample, *, external_branch_drives):
        self._external_context(sample, external_branch_drives)

    def project_current(self, sample, *, external_branch_drives=()):
        nodal, eta, access, _, external = self._parts(sample)
        self._external_context(sample, external_branch_drives)
        total_eta = np.bincount(self._group, weights=eta, minlength=4096)
        result = np.zeros(len(self.state_ids))
        with np.errstate(over="raise", invalid="raise"):
            try:
                projected = eta - self._p * total_eta[self._group] + self._b * access[self._group]
                np.add.at(result, self._nodes, projected)
                np.add.at(result, self._positive, nodal)
                has_negative = self._negative >= 0
                np.add.at(result, self._negative[has_negative], -nodal[has_negative])
                np.add.at(result, self._external_nodes, external)
            except FloatingPointError as error:
                raise ValueError("physical noise nodal current must be representable") from error
        if not np.all(np.isfinite(result)):
            raise ValueError("physical noise nodal current must be finite")
        return result

    def amplifier_input_volts(self, sample):
        _, _, _, voltage, _ = self._parts(sample)
        return None if self.amplifier_input_noise_role is None else tuple(float(value) for value in voltage)

    def work(self, sample, *, mean_state_volts, field_port_volts, duration_seconds,
             external_branch_drives=()):
        nodal, eta, access, _, external = self._parts(sample)
        external_volts = self._external_context(sample, external_branch_drives)
        if np.iscomplexobj(mean_state_volts) or np.iscomplexobj(field_port_volts):
            raise ValueError("physical noise work requires real voltages")
        mean, field = np.asarray(mean_state_volts, dtype=float), np.asarray(field_port_volts, dtype=float)
        duration = _number(duration_seconds, "duration_seconds")
        if (duration <= 0 or mean.shape != (len(self.state_ids),) or field.shape != (4096,)
                or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(field))):
            raise ValueError("noise work requires finite complete means/field ports and positive duration")
        total_eta = np.bincount(self._group, weights=eta, minlength=4096)
        with np.errstate(over="raise", invalid="raise"):
            try:
                mean_v = np.bincount(self._group, weights=self._fractions * mean[self._nodes], minlength=4096)
                junction = (field + access) * self._inverse_d + self._mix * mean_v - self._r_div_d * total_eta
                current = (self._totals * self._inverse_d) * (field + access - mean_v) + total_eta * self._inverse_d
                correction = (self._totals * self._inverse_d) * access + total_eta * self._inverse_d
                drops = mean[self._positive].copy()
                has_negative = self._negative >= 0
                drops[has_negative] -= mean[self._negative[has_negative]]
                current_work = duration * float(nodal @ drops + eta @ (mean[self._nodes] - junction[self._group]))
                current_work += duration * float(external @ (mean[self._external_nodes] - external_volts))
                access_work = duration * float(access @ current)
                field_correction = duration * float(field @ correction)
                external_correction = duration * float(external @ external_volts)
            except FloatingPointError as error:
                raise ValueError("physical noise port work must be representable") from error
        if not all(math.isfinite(value) for value in (current_work, access_work, field_correction, external_correction)):
            raise ValueError("physical noise port work must be finite")
        return NoiseWork(current_work, access_work, field_correction, external_correction)
