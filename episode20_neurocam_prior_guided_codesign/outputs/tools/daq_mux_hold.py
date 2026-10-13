"""Explicit passive mux/hold mechanism, not a reconstructed NI front end.

Each physical ADC group retains one capacitor node. Its source switches are
bidirectional resistors in the same solve as the array: loading and retained
charge affect the source as well as the hold node. No source interval mean is
substituted for the time-varying voltage that drives this node.

All capacitances, initial voltages, track/off/leak conductances and source
membership are caller-supplied. Explicit ideal external voltage ports can
drive the same hold nodes through declared passive switch paths; these are
separate from the 64 neural source lines. This finite mechanism includes no
active buffer, extra filter, saturation/recovery, noise spectrum or actual ADC
aperture/configuration. Such mechanisms must not be assumed absent from
NeuroCam merely because they are absent from this component.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be an explicit finite real number")
    return float(value)


def external_daq_branch_id(adc_id, input_id):
    """Unambiguous identity for one assembled external-input switch path."""
    if any(not isinstance(value, str) or not value.strip() for value in (adc_id, input_id)):
        raise ValueError("external branch identity needs explicit ADC and input names")
    return f"daq_external:{len(adc_id)}:{adc_id}:{len(input_id)}:{input_id}"


@dataclass(frozen=True)
class ExternalDAQBranch:
    branch_id: str
    adc_id: str
    input_id: str
    hold_state_id: str
    track_conductance_siemens: float
    off_conductance_siemens: float

    def __post_init__(self):
        if (self.branch_id != external_daq_branch_id(self.adc_id, self.input_id)
                or not isinstance(self.hold_state_id, str) or not self.hold_state_id.strip()):
            raise ValueError("external branch must retain canonical identity and its explicit hold endpoint")
        track = _number(self.track_conductance_siemens, "external branch track conductance")
        off = _number(self.off_conductance_siemens, "external branch off conductance")
        if track <= 0 or off < 0:
            raise ValueError("external branch needs positive track and nonnegative off conductance")
        object.__setattr__(self, "track_conductance_siemens", track)
        object.__setattr__(self, "off_conductance_siemens", off)


@dataclass(frozen=True)
class ExternalDAQBranchDrive:
    adc_id: str
    input_id: str
    hold_state: int
    conductance_siemens: float
    source_volts: float
    voltage_reference_id: str

    def __post_init__(self):
        external_daq_branch_id(self.adc_id, self.input_id)
        if type(self.hold_state) is not int or self.hold_state < 0:
            raise ValueError("external branch hold must be an explicit state index")
        if not isinstance(self.voltage_reference_id, str) or not self.voltage_reference_id.strip():
            raise ValueError("external branch voltages need an explicit common reference")
        conductance = _number(self.conductance_siemens, "external branch conductance")
        if conductance < 0:
            raise ValueError("external branch conductance must be nonnegative")
        object.__setattr__(self, "conductance_siemens", conductance)
        object.__setattr__(self, "source_volts", _number(self.source_volts, "external source voltage"))

    @property
    def branch_id(self):
        return external_daq_branch_id(self.adc_id, self.input_id)


@dataclass(frozen=True)
class ExternalVoltageInput:
    """One passive mux path to a named ideal external voltage source.

    input_id names the same physical source if reused by another ADC group.
    Conductances describe this path only. Neither voltage nor source calibration
    is inferred here, and the port is not an additional neural array source.
    """
    input_id: str
    track_conductance_siemens: float
    off_conductance_siemens: float

    def __post_init__(self):
        if not isinstance(self.input_id, str) or not self.input_id.strip():
            raise ValueError("external input_id must name a physical voltage source")
        track = _number(self.track_conductance_siemens, "external track conductance")
        off = _number(self.off_conductance_siemens, "external off conductance")
        if track <= 0 or off < 0:
            raise ValueError("external inputs require positive track and nonnegative off conductance")
        object.__setattr__(self, "track_conductance_siemens", track)
        object.__setattr__(self, "off_conductance_siemens", off)


@dataclass(frozen=True)
class DAQGroupDrive:
    """Owned interval values and selection for a group with external inputs.

    Every declared port's voltage is supplied even when unselected: its off
    path can still load the hold node. All values use the named common voltage
    reference. The surrounding physical schedule owns every voltage boundary;
    this object does not evaluate a waveform or select a sampling convention.
    """
    selected_input: int | str | None
    external_volts: tuple
    voltage_reference_id: str

    def __post_init__(self):
        selected = self.selected_input
        if (selected is not None and not (type(selected) is int and selected >= 0)
                and not (isinstance(selected, str) and bool(selected.strip()))):
            raise ValueError("selected_input must be a source index, named external input or None")
        if not isinstance(self.voltage_reference_id, str) or not self.voltage_reference_id.strip():
            raise ValueError("external drive requires an explicit voltage reference")
        values, names = [], set()
        for pair in self.external_volts:
            pair = tuple(pair)
            if (len(pair) != 2 or not isinstance(pair[0], str) or not pair[0].strip()
                    or pair[0] in names):
                raise ValueError("external voltages need unique named input/value pairs")
            names.add(pair[0])
            values.append((pair[0], _number(pair[1], "external input voltage")))
        object.__setattr__(self, "external_volts", tuple(values))


@dataclass(frozen=True)
class PassiveMuxHoldGroup:
    adc_id: str
    source_indices: tuple
    hold_capacitance_farad: float
    initial_hold_volts: float
    track_conductance_siemens: tuple
    off_conductance_siemens: tuple
    hold_leak_conductance_siemens: float
    external_inputs: tuple = ()

    def __post_init__(self):
        if not isinstance(self.adc_id, str) or not self.adc_id:
            raise ValueError("adc_id must name a physical group")
        sources = tuple(self.source_indices)
        external = tuple(self.external_inputs)
        if (any(not isinstance(port, ExternalVoltageInput) for port in external)
                or len({port.input_id for port in external}) != len(external)):
            raise ValueError("external inputs must be distinct declared voltage ports per ADC group")
        if (not sources and not external
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError("Supply distinct nonnegative source_indices or nonempty external inputs")
        track = tuple(_number(g, "track conductance") for g in self.track_conductance_siemens)
        off = tuple(_number(g, "off conductance") for g in self.off_conductance_siemens)
        capacitance = _number(self.hold_capacitance_farad, "hold capacitance")
        initial = _number(self.initial_hold_volts, "initial hold voltage")
        leak = _number(self.hold_leak_conductance_siemens, "hold leak conductance")
        if (len(track) != len(sources) or len(off) != len(sources)
                or any(g <= 0 for g in track) or any(g < 0 for g in off)
                or capacitance <= 0 or leak < 0):
            raise ValueError("Each input needs positive track/nonnegative off G, positive hold C and nonnegative leak G")
        for name, value in (("source_indices", sources), ("track_conductance_siemens", track),
                            ("off_conductance_siemens", off), ("hold_capacitance_farad", capacitance),
                            ("initial_hold_volts", initial), ("hold_leak_conductance_siemens", leak),
                            ("external_inputs", external)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PassiveDAQFrontend:
    """Declared topology and load ownership for a finite passive mechanism.

    source_load_role must be 'additional_fixed_shunt_excludes_frontend': the
    array's existing source_load_conductance_siemens remains an additional
    fixed load and must exclude every load represented here. No ownership mode
    silently deletes a mux path or replaces its dynamic load by a scalar.
    The hold capacitances/leaks use the array's explicit voltage reference.
    Reusing a source in two groups represents two physical parallel loads.
    """
    groups: tuple
    source_load_role: str

    def __post_init__(self):
        groups = tuple(self.groups)
        if (not groups or any(not isinstance(group, PassiveMuxHoldGroup) for group in groups)
                or len({group.adc_id for group in groups}) != len(groups)):
            raise ValueError("Supply nonempty groups with unique physical ADC identities")
        if self.source_load_role != "additional_fixed_shunt_excludes_frontend":
            raise ValueError("The existing fixed source shunt must explicitly exclude this frontend")
        object.__setattr__(self, "groups", groups)

    @property
    def hold_state_count(self):
        return len(self.groups)

    @property
    def switch_branch_count(self):
        """Declared physical switch paths, including presently zero/off paths."""
        return sum(len(group.source_indices) + len(group.external_inputs) for group in self.groups)

    def validate_source_count(self, count):
        if any(source >= count for group in self.groups for source in group.source_indices):
            raise ValueError("DAQ source membership exceeds the assembled array")

    def _selections(self, selected_sources):
        """Normalize selectors without interpreting external ports as states."""
        if (not isinstance(selected_sources, Mapping)
                or set(selected_sources) != {group.adc_id for group in self.groups}):
            raise ValueError("Every physical DAQ group needs an explicit selection")
        normalized = []
        for group in self.groups:
            declaration = selected_sources[group.adc_id]
            if group.external_inputs:
                if not isinstance(declaration, DAQGroupDrive):
                    raise ValueError("Groups with external inputs require a complete DAQGroupDrive")
                selected = declaration.selected_input
                values = dict(declaration.external_volts)
                names = {port.input_id for port in group.external_inputs}
                if set(values) != names:
                    raise ValueError("Supply every declared external voltage, including all off paths")
                valid_external = isinstance(selected, str) and selected in names
            else:
                selected, values, valid_external = declaration, {}, False
            if (selected is not None and not valid_external
                    and (isinstance(selected, bool) or not isinstance(selected, int)
                         or selected not in group.source_indices)):
                raise ValueError("Select a group member source/external input or explicit None for all-off/hold")
            normalized.append((group, selected, values, declaration))
        return tuple(normalized)

    def resistor_edges(self, *, selected_sources, source_node_offset, first_hold_state):
        """Return only the reciprocal neural-source-to-hold resistor edges.

        Selection changes conductance only, not capacitance or retained voltage.
        Off conductance remains present during hold. Additional switch charge
        must be supplied separately in the full array's explicit charge vector.
        An external selection leaves every neural path off. The assembler must
        additionally consume external_drives for the external forcing and load;
        these known-voltage ports cannot be encoded as internal resistor edges.
        """
        edges = []
        for index, (group, selected, _, _) in enumerate(self._selections(selected_sources)):
            hold = first_hold_state + index
            for source, track, off in zip(group.source_indices, group.track_conductance_siemens,
                                           group.off_conductance_siemens):
                conductance = track if source == selected else off
                if conductance > 0:
                    edges.append((source_node_offset + source, hold, conductance))
        return tuple(edges)

    def external_drives(self, *, selected_sources, first_hold_state, voltage_reference_id):
        """Return (hold_node, conductance, known_volts), including off loads.

        The common-reference assembler stamps each branch as diagonal +g and
        forcing +g*u. Physical source work is dt*g*u*(u-mean_hold_volts),
        distinct from kernel nodal-current work. These declarations add no
        state, capacitor, conversion, voltage reset or implicit boundary charge.
        All shared input IDs must carry identical voltages across ADC groups.
        """
        return tuple((branch.hold_state, branch.conductance_siemens, branch.source_volts)
                     for branch in self._external_branch_values(selected_sources=selected_sources,
                         first_hold_state=first_hold_state, voltage_reference_id=voltage_reference_id)
                     if branch.conductance_siemens > 0)

    def external_branch_inventory(self, *, state_ids, first_hold_state, max_branches):
        """All actual external paths, including zero-off paths, in group order."""
        if (type(max_branches) is not int or max_branches < 1
                or type(first_hold_state) is not int or first_hold_state < 0
                or first_hold_state + len(self.groups) > len(state_ids)):
            raise ValueError("complete retained hold-state basis is required")
        result = []
        for index, group in enumerate(self.groups):
            for port in group.external_inputs:
                if len(result) >= max_branches:
                    raise ValueError("external branch inventory exceeds its finite count limit")
                result.append(ExternalDAQBranch(external_daq_branch_id(group.adc_id, port.input_id),
                    group.adc_id, port.input_id, state_ids[first_hold_state + index],
                    port.track_conductance_siemens, port.off_conductance_siemens))
        return tuple(result)

    def external_branch_drives(self, *, selected_sources, first_hold_state, voltage_reference_id):
        """Identified current interval paths; zero-conductance branches remain.

        External branch noise must use this actual track/off conductance and
        common-reference U, rather than treating the remote ideal source as
        ground. This method adds no branch or ADC beyond the declared frontend.
        """
        return tuple(self._external_branch_values(selected_sources=selected_sources,
            first_hold_state=first_hold_state, voltage_reference_id=voltage_reference_id))

    def _external_branch_values(self, *, selected_sources, first_hold_state, voltage_reference_id):
        if (type(first_hold_state) is not int or first_hold_state < 0
                or not isinstance(voltage_reference_id, str) or not voltage_reference_id.strip()):
            raise ValueError("explicit hold-state offset and common voltage reference required")
        shared = {}
        for index, (group, selected, values, declaration) in enumerate(self._selections(selected_sources)):
            if not group.external_inputs:
                continue
            if declaration.voltage_reference_id != voltage_reference_id:
                raise ValueError("external drive voltage reference differs from the assembled reference")
            for port in group.external_inputs:
                value = values[port.input_id]
                if port.input_id in shared and shared[port.input_id] != value:
                    raise ValueError("shared external input must have the same voltage in every ADC group")
                shared[port.input_id] = value
                conductance = (port.track_conductance_siemens if selected == port.input_id
                               else port.off_conductance_siemens)
                yield ExternalDAQBranchDrive(group.adc_id, port.input_id,
                    first_hold_state + index, conductance, value, voltage_reference_id)


@dataclass(frozen=True)
class HoldNodeObservation:
    """Analog diagnostics, not a returned/quantized/available digital sample.

    The caller must bind adc_id to actual conversion metadata/input_id and
    split every row, field, mux/hold and aperture boundary before integration.
    """
    adc_id: str
    state_id: str
    mean_volts: float
    end_volts: float
