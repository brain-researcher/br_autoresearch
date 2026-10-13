"""Full retained NeuroCam topology with explicit contact-voltage boundaries.

Inputs are already referenced, pad-averaged voltages in volts. Each interval
holds those supplied voltages constant; the caller chooses physical boundaries
and waveform resolution. No field generator, parameter fit, noise, actual DAQ
timing or independent source model is inferred by this analog assembler.
"""

from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.sparse import coo_matrix

from causal_cpe import CPERealization
from analog_conditioning import AnalogConditioning
from physical_noise_ports import NoiseLimits, NoiseSample, PhysicalNoiseLayout
from polynomial_noise import NoisePolynomial
from polynomial_circuit_forcing import PolynomialCircuitForcing
from radau_polynomial_forcing import RadauPolynomialCircuitForcing
from conditioned_array_solver import ConditionedRCNetwork, ConditionedSolverControls
from daq_mux_hold import HoldNodeObservation, PassiveDAQFrontend
from sparse_array_rc import (PassiveStarConductance, SparsePassiveRCNetwork,
                             _array, _edges)


@dataclass(frozen=True)
class AssemblyLimits:
    max_states: int
    max_capacitance_edges: int
    max_star_terms: int
    max_stored_numeric_bytes: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class AssemblyResources:
    base_states: int
    fractional_contact_states: int
    ideal_series_interface_states: int
    total_states: int
    mutual_capacitance_edges: int
    capacitance_nonzeros: int
    star_terms: int
    stored_numeric_bytes: int
    daq_hold_states: int = 0
    daq_switch_branches: int = 0
    conditioning_stages: int = 0
    conditioning_states: int = 0
    conditioning_drive_numeric_bytes: int = 0
    physical_noise_sources: int = 0
    physical_noise_numeric_bytes: int = 0


@dataclass(frozen=True)
class DrivenArrayInterval:
    """Physical storage/port work are distinct from kernel Norton bookkeeping.

    kernel_interval retains its surrogate grounded-capacitor charge/energy and
    nodal current work. The physical fields below restore driven-capacitor
    energy and voltage-source work. Boundary work assumes voltage and injected
    charge follow a common linear, zero-duration ramp. These are represented
    circuit quantities, not complete device switching power or thermal cost.
    """
    kernel_interval: object
    voltage_reference_id: str
    physical_energy_before_boundary_joule: float
    physical_energy_after_boundary_joule: float
    physical_energy_end_joule: float
    field_source_boundary_work_joule: float
    supplied_charge_boundary_work_joule: float
    field_source_interval_work_joule: float
    daq_hold_observations: tuple = ()
    daq_external_source_interval_work_joule: float = 0.0
    controlled_voltage_source_interval_work_joule: float = 0.0
    conditioning_drive_branch_dissipation_joule: float = 0.0
    noise_current_source_interval_work_joule: float = 0.0
    noise_access_voltage_source_interval_work_joule: float = 0.0


class NeurocamCPEArray:
    """4096 pads + 64 sources + every required contact dynamic state.

    Fractional contacts append K capacitor states and exactly eliminate only
    their massless access junction. Ideal contacts with R>0 append one dynamic
    interface state. Ideal contacts with R=0 use a driven mutual capacitor,
    represented by added diagonal C plus C*delta(U) boundary charge. Its
    physical energy and source work are restored explicitly.

    Initial contact-state voltages are flattened in row-major pad order, then
    branch order; an ideal R>0 contact contributes its single interface voltage.
    No branch state, off-gate pad or unread source is dropped across row changes.
    All access resistances are supplied at their actual pad area, exactly once.
    With access_resistance_owner='upstream', the supplied voltage must already
    be the solved upstream interface voltage; all added R values must be zero.
    This is not permission to call an unloaded field voltage a loaded interface.
    Reported source work excludes whatever circuit remains upstream of that port.

    An optional explicit PassiveDAQFrontend appends one retained hold capacitor
    per physical ADC group. Its bidirectional mux paths participate in this
    same network. Existing fixed source loads must exclude that frontend.
    External voltage inputs add their declared finite-conductance paths to
    those same hold states, with explicit common-reference voltages and separate
    source work. They add neither neural source lines nor instantaneous charge.
    None retains the earlier array-only model; it asserts no actual DAQ fact.

    Optional AnalogConditioning appends all 64 buffer states then all 64 filter
    states before those hold states. Its explicit input G/C load the original
    sources once; neural mux inputs connect to the filters. The controlled drive,
    loaded recovery, CPE contacts and holds advance jointly. Original source
    observations remain diagnostics. Internal drive rails are not output clamps
    and controlled voltage-source terminal work is not total supply consumption.

    AssemblyLimits bounds state/branch counts before assembly and retained
    NumPy/sparse arrays including sparse LU after assembly. stored_numeric_bytes
    excludes Python scalar/tuple/dataclass objects, including frontend parameter
    objects. It does not bound peak process memory or temporary factorization/
    exponential workspace; the enclosing Slurm memory limit is still required. No throughput
    or memory claim transfers from the earlier 4160-state RC fixture.
    """
    ROWS = 64
    SOURCES = 64
    PADS = 4096
    BASE_STATES = 4160

    def __init__(self, *, contacts, access_resistance_ohm, access_resistance_owner,
                 voltage_reference_id, initial_field_volts, initial_pad_volts,
                 initial_source_volts, initial_contact_state_volts,
                 pad_capacitance_farad, source_capacitance_farad, mutual_capacitance_edges,
                 tft_on_conductance_siemens, tft_off_conductance_siemens,
                 pad_leak_conductance_siemens, source_load_conductance_siemens,
                 other_resistor_edges, limits, daq_frontend=None, analog_conditioning=None):
        if not isinstance(limits, AssemblyLimits):
            raise ValueError("explicit AssemblyLimits are required")
        self._assembly_limits = limits
        self._physical_noise_layout = None
        self._advanced = False
        if not isinstance(voltage_reference_id, str) or not voltage_reference_id:
            raise ValueError("a nonempty common voltage_reference_id is required")
        contacts = tuple(contacts)
        if len(contacts) != self.PADS or any(not isinstance(c, CPERealization) for c in contacts):
            raise ValueError("one explicit CPERealization is required for each of the 4096 pads")
        shape = (self.ROWS, self.SOURCES)
        access = _array("access_resistance_ohm", access_resistance_ohm, shape).ravel()
        if np.any(access < 0) or access_resistance_owner not in ("assembler", "upstream"):
            raise ValueError("nonnegative access resistance and owner 'assembler' or 'upstream' are required")
        if access_resistance_owner == "upstream" and np.any(access != 0):
            raise ValueError("upstream-owned access resistance cannot be added again here")
        fractional = sum(c.internal_state_count for c in contacts if c.alpha < 1)
        ideal_series = sum(int(c.alpha == 1 and resistance > 0) for c, resistance in zip(contacts, access))
        if daq_frontend is not None and not isinstance(daq_frontend, PassiveDAQFrontend):
            raise ValueError("daq_frontend must be an explicit PassiveDAQFrontend or None")
        if daq_frontend is not None:
            daq_frontend.validate_source_count(self.SOURCES)
        daq_groups = () if daq_frontend is None else daq_frontend.groups
        contact_count = fractional + ideal_series
        if analog_conditioning is not None and not isinstance(analog_conditioning, AnalogConditioning):
            raise ValueError("analog_conditioning must be explicit AnalogConditioning or None")
        conditioning_count = 0 if analog_conditioning is None else analog_conditioning.state_count
        conditioning_first = self.BASE_STATES + contact_count
        self._daq_first_state = conditioning_first + conditioning_count
        self._daq_frontend = daq_frontend
        self._analog_conditioning = analog_conditioning
        size = self._daq_first_state + len(daq_groups)
        base_cap_edges = _edges("mutual_capacitance_edges", mutual_capacitance_edges, self.BASE_STATES)
        other_edges = _edges("other_resistor_edges", other_resistor_edges, self.BASE_STATES)
        cap_edge_count = len(base_cap_edges) + fractional + ideal_series
        star_term_count = self.PADS + fractional
        if size > limits.max_states or cap_edge_count > limits.max_capacitance_edges or star_term_count > limits.max_star_terms:
            raise ValueError("declared state, capacitance-edge or star-term assembly limit exceeded")
        stamp = None if analog_conditioning is None else analog_conditioning.stamp(
            state_count=size, source_node_offset=self.PADS,
            first_conditioning_state=conditioning_first, voltage_reference_id=voltage_reference_id)
        self._controlled_drives = None if stamp is None else stamp.controlled_drives
        self._daq_source_offset = self.PADS if stamp is None else conditioning_first + self.SOURCES
        if stamp is not None and stamp.output_nodes != tuple(range(self._daq_source_offset,
                                                                  self._daq_source_offset + self.SOURCES)):
            raise ValueError("conditioning filter states must retain the contiguous 64-source order")
        self._size = size
        self._reference_id = voltage_reference_id
        self._last_field = _array("initial_field_volts", initial_field_volts, shape).ravel()
        self._on = _array("tft_on_conductance_siemens", tft_on_conductance_siemens, shape).ravel()
        self._off = _array("tft_off_conductance_siemens", tft_off_conductance_siemens, shape).ravel()
        if np.any(self._on <= 0) or np.any(self._off <= 0):
            raise ValueError("every pad requires explicit positive finite on and off conductances")
        base_ground = np.concatenate((_array("pad_capacitance_farad", pad_capacitance_farad, shape).ravel(),
                                      _array("source_capacitance_farad", source_capacitance_farad, (self.SOURCES,)),
                                      np.zeros(contact_count),
                                      np.array(() if stamp is None else stamp.added_capacitance_farad),
                                      np.array([group.hold_capacitance_farad for group in daq_groups])))
        self._shunts = np.concatenate((_array("pad_leak_conductance_siemens", pad_leak_conductance_siemens, shape).ravel(),
                                       _array("source_load_conductance_siemens", source_load_conductance_siemens, (self.SOURCES,)),
                                       np.zeros(contact_count),
                                       np.array(() if stamp is None else stamp.added_shunts_siemens),
                                       np.array([group.hold_leak_conductance_siemens for group in daq_groups])))
        if np.any(base_ground < 0) or np.any(self._shunts < 0):
            raise ValueError("explicit ground capacitances and load/leak conductances must be nonnegative")
        if stamp is not None:
            base_ground[self.PADS:self.BASE_STATES] += stamp.source_capacitance_farad
            self._shunts[self.PADS:self.BASE_STATES] += stamp.source_conductance_siemens
            if not np.all(np.isfinite(base_ground)) or not np.all(np.isfinite(self._shunts)):
                raise ValueError("combined conditioning input loads must be finite")
        initial = np.concatenate((_array("initial_pad_volts", initial_pad_volts, shape).ravel(),
                                  _array("initial_source_volts", initial_source_volts, (self.SOURCES,)),
                                  _array("initial_contact_state_volts", initial_contact_state_volts,
                                         (contact_count,)),
                                  np.array(() if stamp is None else stamp.initial_volts),
                                  np.array([group.initial_hold_volts for group in daq_groups])))
        names = [f"pad_{row:02d}_{column:02d}" for row in range(64) for column in range(64)]
        names.extend(f"source_{column:02d}" for column in range(64))
        cap_edges = list(map(tuple, base_cap_edges))
        resistor_edges = list(map(tuple, other_edges))
        groups = []
        retained_contact_noise_edges = []
        self._direct_cap = np.zeros(self.PADS)
        next_state = self.BASE_STATES
        for pad, (contact, resistance) in enumerate(zip(contacts, access)):
            if contact.alpha < 1:
                nodes = [pad]
                weights = [contact.charge_transfer_siemens]
                for branch, (g, c) in enumerate(zip(contact.branch_conductance_siemens,
                                                    contact.branch_capacitance_farad)):
                    nodes.append(next_state)
                    weights.append(g)
                    cap_edges.append((next_state, pad, c))
                    names.append(f"contact_{pad:04d}_branch_{branch}")
                    next_state += 1
                groups.append((tuple(nodes), tuple(weights), float(resistance)))
            elif resistance > 0:
                cap_edges.append((next_state, pad, contact.ideal_capacitance_farad))
                if contact.charge_transfer_siemens:
                    resistor_edges.append((next_state, pad, contact.charge_transfer_siemens))
                    retained_contact_noise_edges.append((f"contact:{pad:04d}:charge_transfer",
                                                          next_state, pad))
                groups.append(((next_state,), (1 / float(resistance),), 0.0))
                names.append(f"contact_{pad:04d}_interface")
                next_state += 1
            else:
                self._direct_cap[pad] = contact.ideal_capacitance_farad
                groups.append(((pad,), (contact.charge_transfer_siemens,), 0.0))
        if next_state != conditioning_first:
            raise ValueError("contact realization state counts do not match their branch data")
        if stamp is not None:
            names.extend(stamp.state_ids)
            resistor_edges.extend(stamp.resistor_edges)
        names.extend(f"daq_hold_{group.adc_id}" for group in daq_groups)
        self._physical_ground = base_ground
        self._cap_edges = _edges("assembled capacitance edges", cap_edges, size)
        self._resistor_edges = _edges("assembled resistor edges", resistor_edges, size)
        self._stars = PassiveStarConductance(state_count=size, groups=groups)
        self._contact_noise_stars = tuple(groups)
        self._retained_contact_noise_edges = tuple(retained_contact_noise_edges)
        effective_ground = base_ground.copy()
        effective_ground[:self.PADS] += self._direct_cap
        network_type = SparsePassiveRCNetwork if stamp is None else ConditionedRCNetwork
        self._network = network_type(state_ids=names, initial_volts=initial,
                                               capacitance_to_reference_farad=effective_ground,
                                               mutual_capacitance_edges=self._cap_edges)
        self._pads = np.arange(self.PADS)
        self._sources = self.PADS + self._pads % self.SOURCES
        self._physical_energy(initial, self._last_field)
        stored_bytes = (self._network.numeric_storage_bytes + self._stars.numeric_storage_bytes
                        + sum(a.nbytes for a in (self._last_field, self._on, self._off, self._physical_ground,
                                                 self._shunts, self._cap_edges, self._resistor_edges,
                                                 self._direct_cap, self._pads, self._sources)))
        conditioning_bytes = 0 if stamp is None else stamp.controlled_drives.numeric_storage_bytes
        stored_bytes += conditioning_bytes
        if stored_bytes > limits.max_stored_numeric_bytes:
            raise ValueError("declared retained numeric-storage limit exceeded")
        self.resources = AssemblyResources(self.BASE_STATES, fractional, ideal_series, size,
                                            len(self._cap_edges), self._network.capacitance_nnz,
                                            self._stars.term_count, stored_bytes, len(daq_groups),
                                            0 if daq_frontend is None else daq_frontend.switch_branch_count,
                                            0 if stamp is None else self.SOURCES,
                                            conditioning_count, conditioning_bytes)

    @property
    def state_ids(self):
        return self._network.state_ids

    @property
    def volts(self):
        return self._network.volts

    @property
    def previous_field_volts(self):
        return tuple(float(v) for v in self._last_field)

    @property
    def voltage_reference_id(self):
        return self._reference_id

    @property
    def daq_frontend(self):
        """Immutable declared frontend metadata; not inverse/policy information."""
        return self._daq_frontend

    @property
    def analog_conditioning(self):
        """Immutable conditional model; None preserves the passive solver path."""
        return self._analog_conditioning

    @property
    def physical_noise_layout(self):
        return self._physical_noise_layout

    def make_noise_layout(self, *, nodal_current_ports, contact_noise_role,
                          amplifier_input_noise_role, excluded_retained_contact_components=(), limits,
                          external_daq_current_ports=()):
        """Bind one explicit noise layout before first physical advancement.

        The actual assembled stars supply topology. In ideal alpha=1 contacts
        with positive access R, their parallel charge-transfer resistor is a
        separate retained edge, not part of the star. Each such positive-G edge
        must have an explicitly owned nodal current port or an explicit reason
        for omission. An omission never establishes physical noise absence.
        External-DAQ current ports bind to this frontend's complete actual
        branch inventory and retain their remote voltage terminal. They append
        amplitudes without changing existing contact or amplifier coordinates.
        """
        if self._physical_noise_layout is not None or self._advanced:
            raise ValueError("noise layout must be fixed once before first physical advance")
        if not isinstance(limits, NoiseLimits):
            raise ValueError("explicit physical noise limits are required")
        if amplifier_input_noise_role is not None and self._controlled_drives is None:
            raise ValueError("amplifier input voltage noise requires assembled conditioning")
        external_inventory = (() if self._daq_frontend is None else
            self._daq_frontend.external_branch_inventory(state_ids=self.state_ids,
                first_hold_state=self._daq_first_state, max_branches=limits.max_sources))
        layout = PhysicalNoiseLayout(state_ids=self.state_ids, voltage_reference_id=self._reference_id,
            nodal_current_ports=nodal_current_ports, contact_stars=self._contact_noise_stars,
            contact_ownership="all_eliminated_star_branch_and_access_noise_owned_here",
            contact_noise_role=contact_noise_role, amplifier_input_noise_role=amplifier_input_noise_role,
            excluded_retained_contact_components=excluded_retained_contact_components, limits=limits,
            external_daq_branches=external_inventory, external_daq_current_ports=external_daq_current_ports)
        owned = {port.owned_component_id: port for port in layout.nodal_current_ports}
        excluded = dict(layout.excluded_retained_contact_components)
        required = {component for component, _, _ in self._retained_contact_noise_edges}
        if set(excluded) - required or set(excluded) & set(owned):
            raise ValueError("retained contact noise omissions must be exact and cannot also be owned")
        for component, first, second in self._retained_contact_noise_edges:
            if component in excluded:
                continue
            port = owned.get(component)
            if port is None or {port.positive_state_id, port.negative_state_id} != {
                    self.state_ids[first], self.state_ids[second]}:
                raise ValueError("retained charge-transfer noise needs its exact nodal port or explicit omission")
        stored = self.resources.stored_numeric_bytes + layout.retained_numeric_bytes
        if stored > self._assembly_limits.max_stored_numeric_bytes:
            raise ValueError("combined array/noise retained numeric-storage limit exceeded")
        self.resources = replace(self.resources, stored_numeric_bytes=stored,
            physical_noise_sources=layout.numeric_value_count,
            physical_noise_numeric_bytes=layout.retained_numeric_bytes)
        self._physical_noise_layout = layout
        return layout

    def validate_noise_sample(self, sample):
        if self._physical_noise_layout is None:
            if sample is not None:
                raise ValueError("noise sample requires the array's explicitly fixed physical layout")
        elif (not isinstance(sample, (NoiseSample, NoisePolynomial)) or sample.layout is not self._physical_noise_layout
              or sample.state_ids != self.state_ids or sample.voltage_reference_id != self._reference_id):
            raise ValueError("noise sample must match the exact fixed array layout, states and reference")
        if isinstance(sample, NoisePolynomial) and self._controlled_drives is None:
            raise ValueError("polynomial noise requires the joint conditioned solver")

    def _validate_external_noise(self, sample, context):
        if isinstance(sample, NoisePolynomial):
            # Check every original coefficient, including a nonconstant term
            # that happens to vanish at the selected midpoint.
            for row in sample.numeric_array:
                self._physical_noise_layout.validate_external_branch_sample(
                    self._physical_noise_layout.sample_from_values(row), external_branch_drives=context)
        else:
            self._physical_noise_layout.validate_external_branch_sample(sample, external_branch_drives=context)

    def validate_noise_interval(self, sample, *, daq_selected_sources):
        """Preflight a held sample or all cubic coefficients against actual paths.

        The executor calls this during its complete-plan traversal, before any
        electrical advancement. A zero-conductance branch requires zero noise
        current; no external ideal source is replaced by the reference node.
        """
        self.validate_noise_sample(sample)
        if sample is not None and self._physical_noise_layout.external_daq_current_ports:
            context = self._daq_frontend.external_branch_drives(
                selected_sources=daq_selected_sources, first_hold_state=self._daq_first_state,
                voltage_reference_id=self._reference_id)
            self._validate_external_noise(sample, context)

    def _physical_energy(self, volts, field):
        # Equivalent to surrogate energy - sum C*U*Vpad + .5 sum C*U^2,
        # but sum the actual nonnegative capacitor energies to avoid cancellation.
        edges = self._cap_edges
        differences = volts[edges[:, 0].astype(int)] - volts[edges[:, 1].astype(int)]
        energy = float(0.5 * (self._physical_ground @ (volts * volts)
                             + edges[:, 2] @ (differences * differences)
                             + self._direct_cap @ ((volts[:self.PADS] - field) ** 2)))
        if not math.isfinite(energy):
            raise ValueError("physical stored energy must be finite")
        return energy

    def advance_row(self, *, gate_row, duration_seconds, field_volts, voltage_reference_id,
                    observed_sources, injected_charge_coulomb, controls, daq_selected_sources=None,
                    check=None, noise_sample=None, interval_start_seconds=None):
        """Advance all states through a caller-chosen constant-voltage interval.

        A field change and supplied charge follow the same linear instantaneous
        ramp before propagation. Previous field voltage is committed only after
        the full numerical interval and physical accounting succeed. Split at
        every physical waveform, row, DAQ selection/hold, parameter or aperture
        change. With a frontend, every group requires an explicit selected
        source or None, or a DAQGroupDrive when it has external inputs. External
        voltages remain specified through off paths and are constant during
        this interval; their changes through finite resistance inject no
        boundary charge. Mean outputs are analog interval means,
        not a claimed ADC convention. Source observations are diagnostics and
        need not be retained digital samples. All states advance either way.

        A NoisePolynomial additionally requires the absolute interval start and
        joint conditioning. Its original support contains this whole interval;
        internal midpoint or explicitly selected Radau stages use that same
        polynomial and their actual stage-state physical work. Radau's final
        stage uses this interval's left limit before the next physical change.
        Held samples keep their original physical representation.

        This call commits the physical interval. A caller that subsequently
        receives a readout refusal must abort its attempt rather than retrying
        that aperture after physical state has already advanced.
        """
        if voltage_reference_id != self._reference_id:
            raise ValueError("field voltages must use the assembled reference convention")
        self.validate_noise_sample(noise_sample)
        polynomial_noise = isinstance(noise_sample, NoisePolynomial)
        if not polynomial_noise and interval_start_seconds is not None:
            raise ValueError("absolute noise interval start is only used with a polynomial")
        if self._controlled_drives is not None:
            if not isinstance(controls, ConditionedSolverControls) or not callable(check):
                raise ValueError("conditioning requires explicit conditioned controls and resource callback")
            check()
        if isinstance(gate_row, bool) or not isinstance(gate_row, int) or not 0 <= gate_row < self.ROWS:
            raise ValueError("gate_row must select a physical row in 0..63")
        selected = tuple(observed_sources)
        if (not selected or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.SOURCES
                                for i in selected) or len(set(selected)) != len(selected)):
            raise ValueError("observed_sources must name distinct physical source lines")
        external_noise_context = ()
        if self._daq_frontend is None:
            if daq_selected_sources is not None:
                raise ValueError("DAQ selections require an explicitly assembled frontend")
            daq_edges = ()
            external_drives = ()
        else:
            daq_edges = self._daq_frontend.resistor_edges(selected_sources=daq_selected_sources,
                                                         source_node_offset=self._daq_source_offset,
                                                         first_hold_state=self._daq_first_state)
            external_drives = self._daq_frontend.external_drives(
                selected_sources=daq_selected_sources, first_hold_state=self._daq_first_state,
                voltage_reference_id=self._reference_id)
            if noise_sample is not None and self._physical_noise_layout.external_daq_current_ports:
                external_noise_context = self._daq_frontend.external_branch_drives(
                    selected_sources=daq_selected_sources, first_hold_state=self._daq_first_state,
                    voltage_reference_id=self._reference_id)
                self._validate_external_noise(noise_sample, external_noise_context)
        field = _array("field_volts", field_volts, (self.ROWS, self.SOURCES)).ravel()
        supplied_charge = _array("injected_charge_coulomb", injected_charge_coulomb, (self._size,))
        charge = supplied_charge.copy()
        charge[:self.PADS] += self._direct_cap * (field - self._last_field)
        tft = np.where(self._pads // self.SOURCES == gate_row, self._on, self._off)
        edges = np.vstack((np.column_stack((self._pads, self._sources, tft)), self._resistor_edges))
        if daq_edges:
            edges = np.vstack((edges, np.asarray(daq_edges)))
        shunts = self._shunts
        drive = self._stars.drive_current(port_volts=field)
        controlled_drives = self._controlled_drives
        if noise_sample is not None and not polynomial_noise:
            drive += self._physical_noise_layout.project_current(noise_sample,
                external_branch_drives=external_noise_context)
            amplifier_noise = self._physical_noise_layout.amplifier_input_volts(noise_sample)
            if amplifier_noise is not None:
                controlled_drives = controlled_drives.with_input_noise_volts(amplifier_noise)
        if external_drives:
            shunts = shunts.copy()
            for node, conductance, voltage in external_drives:
                shunts[node] += conductance
                drive[node] += conductance * voltage
        observation = coo_matrix((np.ones(len(selected)),
                                  (np.arange(len(selected)), self.PADS + np.array(selected))),
                                 shape=(len(selected), self._size)).tocsr()
        conditioning_arguments = {} if self._controlled_drives is None else dict(
            controlled_drives=controlled_drives, check=check)
        if polynomial_noise:
            forcing_type = (RadauPolynomialCircuitForcing if controls.integration_method == "radau_iia_2"
                            else PolynomialCircuitForcing)
            conditioning_arguments["time_varying_forcing"] = forcing_type(noise_sample,
                interval_start_seconds=interval_start_seconds, duration_seconds=duration_seconds,
                base_controlled_drives=controlled_drives, field_port_volts=field,
                external_branch_drives=external_noise_context,
                max_stored_numeric_bytes=(self._assembly_limits.max_stored_numeric_bytes
                                          - self.resources.stored_numeric_bytes), check=check)
        result = self._network.advance(duration_seconds=duration_seconds,
                                        conductance_to_reference_siemens=shunts,
                                        resistor_edges=edges, passive_stars=self._stars,
                                        drive_current_amp=drive,
                                        observation_matrix=observation, feedthrough_volts=np.zeros(len(selected)),
                                        injected_charge_coulomb=charge, controls=controls, commit=False,
                                        **conditioning_arguments)
        before = np.array(result.boundary.transition.before_volts)
        start = np.array(result.observation.start_volts)
        end = np.array(result.observation.end_volts)
        mean = np.array(result.observation.mean_volts)
        source_charge_change = self._direct_cap * ((field - start[:self.PADS])
                                                   - (self._last_field - before[:self.PADS]))
        boundary_work = float(0.5 * ((self._last_field + field) @ source_charge_change))
        supplied_work = float(0.5 * ((before + start) @ supplied_charge))
        interval_work = (self._stars.source_work(port_volts=field, mean_state_volts=mean,
                                                 duration_seconds=duration_seconds)
                         - float((field * self._direct_cap) @ (end[:self.PADS] - start[:self.PADS])))
        external_work = float(duration_seconds) * math.fsum(
            conductance * voltage * (voltage - float(mean[node]))
            for node, conductance, voltage in external_drives)
        controlled_work = (0.0 if self._controlled_drives is None
                           else result.controlled_voltage_source_work_joule)
        controlled_heat = (0.0 if self._controlled_drives is None
                           else result.controlled_branch_dissipation_joule)
        current_noise_work = access_noise_work = 0.0
        if polynomial_noise:
            # Solver integrals use original-port amplitudes at actual internal
            # midpoint/Radau stage states; multiplying interval means would
            # lose their covariance. Original physical work slots are unchanged.
            current_noise_work, access_noise_work, field_correction, external_correction = result.physical_noise_work_joule
            interval_work += field_correction
            external_work += external_correction
        elif noise_sample is not None:
            noise_work = self._physical_noise_layout.work(noise_sample, mean_state_volts=mean,
                field_port_volts=field, duration_seconds=duration_seconds,
                external_branch_drives=external_noise_context)
            interval_work += noise_work.field_source_interval_work_correction_joule
            external_work += noise_work.daq_external_source_interval_work_correction_joule
            current_noise_work = noise_work.noise_current_source_interval_work_joule
            access_noise_work = noise_work.noise_access_voltage_source_interval_work_joule
        if not all(math.isfinite(v) for v in (boundary_work, supplied_work, interval_work, external_work,
                                             controlled_work, controlled_heat, current_noise_work, access_noise_work)):
            raise ValueError("physical boundary and interval source work must be finite")
        physical = DrivenArrayInterval(result, self._reference_id,
                                        self._physical_energy(before, self._last_field),
                                        self._physical_energy(start, field), self._physical_energy(end, field),
                                        boundary_work, supplied_work, interval_work,
                                        tuple(HoldNodeObservation(group.adc_id,
                                                                 self.state_ids[self._daq_first_state + index],
                                                                 float(mean[self._daq_first_state + index]),
                                                                 float(end[self._daq_first_state + index]))
                                              for index, group in enumerate(
                                                  () if self._daq_frontend is None else self._daq_frontend.groups)),
                                        external_work, controlled_work, controlled_heat,
                                        current_noise_work, access_noise_work)
        self._network.commit_preview(result)
        self._last_field = field
        self._advanced = True
        return physical
