"""One prospective full-size R3 cost scenario, not a fitted NeuroCam model.

Build parameters, not an extra array. The production wrapper instantiates the
one persistent array and calls build_conditional_noise on that same instance.
All 4,096 fractional contacts, 64 buffers/filters and 24 ADC holds are retained.
No data access, schedule, launch, candidate scoring or parameter tuning occurs
here. Numerical execution caps belong to the separately explicit run recipe.
"""

from dataclasses import asdict
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from analog_conditioning import AnalogConditioning, SourceConditioning
from causal_cpe import realize_contact_cpe
from circuit_noise_provider import SpectralCircuitNoise, SpectralCubicCircuitNoise
from cpe_array import AssemblyLimits, NeurocamCPEArray
from daq_mux_hold import PassiveDAQFrontend
from passive_contact import BOLTZMANN_J_PER_K, ContactParameters
from physical_noise_ports import NoiseLimits, NodalCurrentPort
from spectral_noise import (SpectralNoise, SpectralNoiseLimits, LatentQuadrature,
                            draw_standard_normal_quadratures)


RECIPE_ID = "conditional_R3_B2_2001_G0_001"
REFERENCE_ID = "conditional_R3_zero_reference"
ADC_IDS = tuple(f"conditional_adc_{index:02d}" for index in range(24))
CONTACT_BRANCHES = 16
STATE_COUNT = 69848
TEMPERATURE_K = 300.0
NOISE_SEED = 202010092001
NOISE_BIN_EDGES_HZ = (0.125, 0.5, 2., 8., 32., 128., 512., 2048., 8192., 32768.)
RETAINED_NOISE_FILE_CAP = 256 * 1024**2

# All values below are deliberately prospective engineering assumptions.
CONTACT = dict(r_access_l_ohm=2000., g_ct_l_siemens=5e-9,
               q_l_siemens_seconds_alpha=2e-8, alpha=0.5, gamma=1.0)
CONDITIONING = dict(gain=8., offset_volts=0., drive_lower_volts=-0.05,
    drive_upper_volts=0.05, input_capacitance_farad=5e-12,
    input_conductance_siemens=1e-9, buffer_capacitance_farad=100e-12,
    buffer_conductance_siemens=2e-5, output_conductance_siemens=2e-4,
    filter_capacitance_farad=2.2e-9, filter_leak_conductance_siemens=1e-9,
    initial_buffer_volts=0., initial_filter_volts=0.)
OMISSIONS = (
    "biased_TFT_excess_and_switching_dependent_noise_not_characterized",
    "neural_and_external_mux_branch_noise_law_not_selected",
    "controlled_drive_extra_noise_beyond_explicit_input_voltage_and_current",
    "unmodeled_supply_noise_efficiency_and_quiescent_power",
    "mechanical_motion_electrochemical_drift_and_population_device_variation",
    "continuous_spectrum_quadrature_accuracy_and_stationary_initial_state_not_established",
)


def metadata():
    """JSON-safe exact constants; no measured noise or performance assertion."""
    return dict(recipe_id=RECIPE_ID, status="prospective_conditional_cost_scenario",
        world_condition="B2", world_replicate_seed=2001, geometry_id="G0",
        pad_area_um2=9450., short_axis="x", duration_seconds=16.,
        voltage_reference_id=REFERENCE_ID, contact=dict(CONTACT),
        contact_branch_count=CONTACT_BRANCHES, total_states=STATE_COUNT,
        cpe_pole_band_rad_per_second=[2 * math.pi * 1e-4, 2 * math.pi * 1e8],
        cpe_forward_band_hz=[0.125, 32768.], cpe_grid_points=257,
        cpe_grid_relative_tolerance=0.05, cpe_truncation_relative_tolerance=0.05,
        pad_capacitance_farad=2e-12, source_capacitance_farad=30e-12,
        horizontal_vertical_pad_mutual_capacitance_farad=0.02e-12,
        adjacent_source_mutual_capacitance_farad=0.2e-12,
        tft_on_conductance_siemens=2e-6, tft_off_conductance_siemens=2e-11,
        pad_leak_conductance_siemens=1e-12, source_load_conductance_siemens=1e-9,
        conditioning=dict(CONDITIONING), initial_all_state_volts=0., initial_field_volts=0.,
        adc_ids=list(ADC_IDS), hold_capacitance_farad=20e-12,
        mux_track_conductance_siemens=1e-3, mux_off_conductance_siemens=1e-10,
        hold_leak_conductance_siemens=1e-10,
        temperature_kelvin=TEMPERATURE_K, noise_bin_edges_hz=list(NOISE_BIN_EDGES_HZ),
        noise_frequency_rule="geometric_bin_centres", noise_weight_rule="bin_width_hz",
        noise_seed=NOISE_SEED, noise_rng="numpy_PCG64_standard_normal_named_canonical_basis",
        amplifier_voltage_asd_v_per_sqrt_hz=8e-9,
        amplifier_current_asd_a_per_sqrt_hz=0.3e-12,
        common_input_reference_asd_v_per_sqrt_hz=2e-9,
        common_reference_scope="same_pre_gain_input_voltage_at_all_64_amplifiers",
        assembly_limits=dict(max_states=70000, max_capacitance_edges=75000,
            max_star_terms=70000, max_stored_numeric_bytes=512 * 1024**2),
        physical_noise_limits=dict(max_sources=80000, max_stored_numeric_bytes=16 * 1024**2),
        retained_noise_file_cap_bytes=RETAINED_NOISE_FILE_CAP,
        omitted_sources=list(OMISSIONS), continuous_white_noise=False,
        fitted_or_calibrated=False, candidate_scoring_authorized=False)


def expected_state_ids():
    return (tuple(f"pad_{row:02d}_{column:02d}" for row in range(64) for column in range(64))
        + tuple(f"source_{source:02d}" for source in range(64))
        + tuple(f"contact_{pad:04d}_branch_{branch}" for pad in range(4096)
                for branch in range(CONTACT_BRANCHES))
        + tuple(f"conditioning_buffer_{source:02d}" for source in range(64))
        + tuple(f"conditioning_filter_{source:02d}" for source in range(64))
        + tuple(f"daq_hold_{adc}" for adc in ADC_IDS))


def _frontend(frontend):
    if (not isinstance(frontend, PassiveDAQFrontend)
            or tuple(group.adc_id for group in frontend.groups) != ADC_IDS):
        raise ValueError("conditional recipe requires its exact 24 ADC groups")
    for group_index, group in enumerate(frontend.groups):
        sources = tuple(source for source in range(64) if source % 24 == group_index)
        if (group.source_indices != sources or group.hold_capacitance_farad != 20e-12
                or group.initial_hold_volts != 0. or group.hold_leak_conductance_siemens != 1e-10
                or group.track_conductance_siemens != (1e-3,) * len(sources)
                or group.off_conductance_siemens != (1e-10,) * len(sources)):
            raise ValueError("frontend differs from selected conditional recipe")
        external = group.external_inputs
        if (len(external) != (1 if group_index == 0 else 0)
                or external and (external[0].input_id != "G1"
                    or external[0].track_conductance_siemens != 1e-3
                    or external[0].off_conductance_siemens != 1e-10)):
            raise ValueError("G1 must share the existing conditional ADC0 hold")


def build_conditional_circuit_parameters(*, daq_frontend, check):
    """Return one complete explicit constructor mapping; never build an array."""
    if not callable(check):
        raise ValueError("bounded execution check required")
    check()
    _frontend(daq_frontend)
    contact = realize_contact_cpe(parameters=ContactParameters(**CONTACT), pad_area_um2=9450.,
        pole_low_rad_per_second=2 * math.pi * 1e-4,
        pole_high_rad_per_second=2 * math.pi * 1e8, branch_count=CONTACT_BRANCHES,
        forward_band_hz=(0.125, 32768.), check_frequencies_hz=np.geomspace(0.125, 32768., 257),
        grid_relative_tolerance=0.05, truncation_relative_tolerance=0.05)
    check()
    edges = []
    for row in range(64):
        check()
        for column in range(64):
            node = row * 64 + column
            if column < 63:
                edges.append((node, node + 1, 0.02e-12))
            if row < 63:
                edges.append((node, node + 64, 0.02e-12))
    edges.extend((4096 + source, 4097 + source, 0.2e-12) for source in range(63))
    conditioning = AnalogConditioning(channels=(SourceConditioning(**CONDITIONING),) * 64,
        source_load_role="additional_fixed_shunt_excludes_conditioning",
        source_capacitance_role="additional_fixed_capacitance_excludes_conditioning",
        voltage_reference_id=REFERENCE_ID)
    result = dict(contacts=(contact,) * 4096,
        access_resistance_ohm=np.full((64, 64), 2000.), access_resistance_owner="assembler",
        voltage_reference_id=REFERENCE_ID, initial_field_volts=np.zeros((64, 64)),
        initial_pad_volts=np.zeros((64, 64)), initial_source_volts=np.zeros(64),
        initial_contact_state_volts=np.zeros(4096 * CONTACT_BRANCHES),
        pad_capacitance_farad=np.full((64, 64), 2e-12),
        source_capacitance_farad=np.full(64, 30e-12), mutual_capacitance_edges=tuple(edges),
        tft_on_conductance_siemens=np.full((64, 64), 2e-6),
        tft_off_conductance_siemens=np.full((64, 64), 2e-11),
        pad_leak_conductance_siemens=np.full((64, 64), 1e-12),
        source_load_conductance_siemens=np.full(64, 1e-9), other_resistor_edges=(),
        limits=AssemblyLimits(max_states=70000, max_capacitance_edges=75000,
            max_star_terms=70000, max_stored_numeric_bytes=512 * 1024**2),
        daq_frontend=daq_frontend, analog_conditioning=conditioning)
    check()
    return result


class _CappedArchive:
    """Bound the one selected NPZ writer; partial failed artifacts are retained."""
    def __init__(self, raw):
        self.raw = raw

    def write(self, value):
        if self.raw.tell() + len(value) > RETAINED_NOISE_FILE_CAP:
            raise ValueError("retained noise archive exceeds 256 MiB")
        return self.raw.write(value)

    def __getattr__(self, name):
        return getattr(self.raw, name)


def _save_noise(path, *, spectrum, tolerances, spectral_limits, check):
    """Save complete named draws and factors before exposing the noise provider."""
    check()
    path = Path(path)
    if path.suffix != ".npz" or not path.parent.is_dir():
        raise ValueError("retained noise path needs an existing directory and .npz suffix")
    terms = np.asarray([(k, p, latent, value.real, value.imag)
                        for k, p, latent, value in spectrum.sparse_terms], dtype=np.float64)
    # All selected indices are < 2**53, so this temporary mixed matrix is exact;
    # the archive stores their three axes as int64, not as float identifiers.
    lookup = {name: index for index, name in enumerate(spectrum.latent_ids)}
    draws = np.asarray([(draw.frequency_index, lookup[draw.latent_id], draw.a, draw.b)
                        for draw in spectrum.quadratures], dtype=np.float64)
    check()
    with path.open("xb") as raw:
        np.savez(_CappedArchive(raw),
            recipe_json=np.asarray(json.dumps(metadata(), sort_keys=True)),
            spectral_limits_json=np.asarray(json.dumps(asdict(spectral_limits), sort_keys=True)),
            frequencies_hz=np.asarray(spectrum.frequencies_hz),
            quadrature_widths_hz=np.asarray(spectrum.quadrature_widths_hz),
            port_ids=np.asarray(spectrum.port_ids), port_units=np.asarray(spectrum.port_units),
            latent_ids=np.asarray(spectrum.latent_ids),
            factor_frequency_index=terms[:, 0].astype(np.int64),
            factor_port_index=terms[:, 1].astype(np.int64),
            factor_latent_index=terms[:, 2].astype(np.int64),
            factor_real=terms[:, 3], factor_imag=terms[:, 4],
            draw_frequency_index=draws[:, 0].astype(np.int64),
            draw_latent_index=draws[:, 1].astype(np.int64),
            draw_a=draws[:, 2], draw_b=draws[:, 3],
            port_absolute_tolerances=np.asarray(tolerances),
            time_window_seconds=np.asarray(spectrum.time_window_seconds))
        raw.flush()
    check()


def _replay_noise(path, *, layout, latents, frequencies, widths, factors, tolerances, limits, check):
    """Load the saved original basis and quadratures without any RNG call.

    Bound NPZ members and their declared NPY arrays before allocating payloads.
    Match the selected recipe/physical basis/factors, so a changed source model
    cannot masquerade as replay of this realization.
    """
    count = len(layout.port_ids)
    term_count, draw_count = 9 * (count + 64), 9 * len(latents)
    specs = {"recipe_json": ((), "U"), "spectral_limits_json": ((), "U"),
        "frequencies_hz": ((9,), "f"), "quadrature_widths_hz": ((9,), "f"),
        "port_ids": ((count,), "U"), "port_units": ((count,), "U"),
        "latent_ids": ((len(latents),), "U"),
        "port_absolute_tolerances": ((count,), "f"), "time_window_seconds": ((2,), "f")}
    for name in ("frequency_index", "port_index", "latent_index", "real", "imag"):
        specs["factor_" + name] = ((term_count,), "i" if name.endswith("index") else "f")
    for name in ("frequency_index", "latent_index", "a", "b"):
        specs["draw_" + name] = ((draw_count,), "i" if name.endswith("index") else "f")
    check()
    with Path(path).open("rb") as raw:
        with zipfile.ZipFile(raw) as archive:
            members = archive.infolist()
            if (len(members) != len(specs) or {m.filename for m in members} != {n + ".npy" for n in specs}
                    or sum(m.file_size for m in members) > RETAINED_NOISE_FILE_CAP):
                raise ValueError("Saved conditional noise archive has an unexpected or oversized basis")
            for member in members:
                check()
                name = member.filename[:-4]
                shape, kind = specs[name]
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        actual, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
                    elif version == (2, 0):
                        actual, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
                    else:
                        raise ValueError("Unsupported saved noise NPY version")
                    string_cap = 65536 if not shape else 1024
                    if (actual != shape or fortran or dtype.kind != kind or dtype.hasobject
                            or (kind != "U" and dtype.itemsize != 8)
                            or (kind == "U" and not 0 < dtype.itemsize <= string_cap)
                            or stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size):
                        raise ValueError("Saved noise array has an unexpected shape, dtype or byte size")
        raw.seek(0)
        with np.load(raw, allow_pickle=False) as saved:
            data = {name: saved[name] for name in specs}
    check()
    if json.loads(str(data["recipe_json"])) != metadata():
        raise ValueError("Saved noise belongs to a different conditional electrical recipe")
    if json.loads(str(data["spectral_limits_json"])) != asdict(limits):
        raise ValueError("Saved noise spectral resource limits differ from this attempt")
    for name, expected in (("frequencies_hz", frequencies), ("quadrature_widths_hz", widths),
            ("port_ids", layout.port_ids), ("port_units", layout.port_units), ("latent_ids", latents),
            ("port_absolute_tolerances", tolerances), ("time_window_seconds", (0., 16.))):
        if not np.array_equal(data[name], np.asarray(expected)):
            raise ValueError(f"Saved noise {name} differs from the selected original basis")
    # Compare factors in their retained order. All named draws are then read
    # directly, including draws unused after projection; never regenerate them.
    for index, (k, p, latent, value) in enumerate(factors()):
        if index % 4096 == 0:
            check()
        if (data["factor_frequency_index"][index] != k or data["factor_port_index"][index] != p
                or data["factor_latent_index"][index] != latent
                or data["factor_real"][index] != complex(value).real
                or data["factor_imag"][index] != complex(value).imag):
            raise ValueError("Saved noise factor differs from the selected physical source model")
    def original_draws():
        for index in range(draw_count):
            if index % 4096 == 0:
                check()
            k, latent = int(data["draw_frequency_index"][index]), int(data["draw_latent_index"][index])
            if not 0 <= k < 9 or not 0 <= latent < len(latents):
                raise ValueError("Saved noise draw lies outside its named frequency/latent basis")
            yield LatentQuadrature(k, latents[latent], float(data["draw_a"][index]), float(data["draw_b"][index]))
    return SpectralNoise(frequencies_hz=frequencies, quadrature_widths_hz=widths,
        port_ids=layout.port_ids, port_units=layout.port_units, latent_ids=latents,
        sparse_terms=factors(), quadratures=original_draws(), time_window_seconds=(0., 16.),
        limits=limits, check=check)


def build_conditional_noise(*, array, check, retained_draw_path,
                            current_hold_tolerance_amp, voltage_hold_tolerance_volts,
                            max_evaluations, max_term_evaluations,
                            reuse_draw_path=None, cubic_cell_for_key=None):
    """Bind this scenario's original ports and save its one unchanged draw.

    Explicit caller-selected A/V tolerances and work caps remain necessary;
    no numerical tolerance is inferred from these stochastic source amplitudes.
    Fixed passive resistors have 4kTG current PSD and access resistors 4kTR
    voltage PSD in the finite conditional band. The nine spectral atoms are
    not claimed to approximate continuous thermal noise to a validated error.
    """
    if (not callable(check) or not isinstance(array, NeurocamCPEArray)
            or array.state_ids != expected_state_ids()
            or array.voltage_reference_id != REFERENCE_ID):
        raise ValueError("the single complete selected conditional array is required")
    _frontend(array.daq_frontend)
    check()
    ports, asd = [], {}
    thermal = 4 * BOLTZMANN_J_PER_K * TEMPERATURE_K

    def add_current(port_id, positive, negative, component, *, conductance=None, amplitude=None):
        check()
        if (conductance is None) == (amplitude is None):
            raise ValueError("one explicit current PSD ownership required")
        role = "300K_finite_band_passive_resistor" if conductance is not None else "conditional_amplifier_current"
        ports.append(NodalCurrentPort(port_id, positive, negative, role, component))
        asd[port_id] = math.sqrt(thermal * conductance) if conductance is not None else amplitude

    for row in range(64):
        for column in range(64):
            pad = row * 64 + column
            add_current(f"pad_leak:{pad:04d}", f"pad_{row:02d}_{column:02d}", None,
                        f"pad_leak:{pad:04d}", conductance=1e-12)
    for source in range(64):
        node = f"source_{source:02d}"
        buffer, filtered = f"conditioning_buffer_{source:02d}", f"conditioning_filter_{source:02d}"
        for label, positive, negative, conductance in (
            ("source_load", node, None, 1e-9),
            ("conditioning_input_load", node, None, 1e-9),
            ("conditioning_output", buffer, filtered, 2e-4),
            ("conditioning_filter_leak", filtered, None, 1e-9)):
            name = f"{label}:{source:02d}"
            add_current(name, positive, negative, name, conductance=conductance)
        name = f"amplifier_current:{source:02d}"
        add_current(name, node, None, name, amplitude=0.3e-12)
    for adc in ADC_IDS:
        name = f"hold_leak:{adc}"
        add_current(name, f"daq_hold_{adc}", None, name, conductance=1e-10)
    layout = array.make_noise_layout(nodal_current_ports=tuple(ports),
        contact_noise_role="300K_finite_band_actual_passive_contact_branches_and_access",
        amplifier_input_noise_role="conditional_independent_and_common_input_voltage",
        limits=NoiseLimits(max_sources=80000, max_stored_numeric_bytes=16 * 1024**2))
    for pad, (_, conductances, access) in enumerate(layout.contact_stars):
        check()
        for branch, conductance in enumerate(conductances):
            asd[f"contact:{pad:04d}:star_branch:{branch}"] = math.sqrt(thermal * conductance)
        asd[f"contact:{pad:04d}:star_access_voltage"] = math.sqrt(thermal * access)
    for source in range(64):
        asd[f"amplifier_input_voltage:{source:02d}"] = 8e-9
    if len(layout.port_ids) != 78232 or set(asd) != set(layout.port_ids):
        raise ValueError("conditional noise inventory differs from the complete selected ports")
    common = "shared:amplifier_input_reference"
    latents = tuple(f"independent:{port}" for port in layout.port_ids) + (common,)
    frequencies = tuple(math.sqrt(left * right) for left, right in zip(NOISE_BIN_EDGES_HZ, NOISE_BIN_EDGES_HZ[1:]))
    widths = tuple(right - left for left, right in zip(NOISE_BIN_EDGES_HZ, NOISE_BIN_EDGES_HZ[1:]))
    limits = SpectralNoiseLimits(max_terms=705000, max_draws=705000, max_frequencies=9,
        max_ports=80000, max_numeric_bytes=128 * 1024**2,
        max_evaluations=max_evaluations, max_term_evaluations=max_term_evaluations)
    def factors():
        for k in range(9):
            for p, port in enumerate(layout.port_ids):
                yield (k, p, p, asd[port])
                if port.startswith("amplifier_input_voltage:"):
                    yield (k, p, len(latents) - 1, 2e-9)
    tolerances = tuple(current_hold_tolerance_amp if unit == "A" else voltage_hold_tolerance_volts
                       for unit in layout.port_units)
    if reuse_draw_path is None:
        if retained_draw_path is None:
            raise ValueError("A new realization requires an explicit retained draw path")
        draws = draw_standard_normal_quadratures(
            frequency_latent_ids=((k, latent) for k in range(9) for latent in latents),
            seed=NOISE_SEED, max_draws=limits.max_draws, check=check)
        spectrum = SpectralNoise(frequencies_hz=frequencies, quadrature_widths_hz=widths,
            port_ids=layout.port_ids, port_units=layout.port_units, latent_ids=latents,
            sparse_terms=factors(), quadratures=draws, time_window_seconds=(0., 16.),
            limits=limits, check=check)
        _save_noise(retained_draw_path, spectrum=spectrum, tolerances=tolerances,
                    spectral_limits=limits, check=check)
    else:
        if retained_draw_path is not None:
            raise ValueError("Replay refers to the retained original archive; do not create a second realization")
        spectrum = _replay_noise(reuse_draw_path, layout=layout, latents=latents,
            frequencies=frequencies, widths=widths, factors=factors,
            tolerances=tolerances, limits=limits, check=check)
    provider_class = SpectralCircuitNoise if cubic_cell_for_key is None else SpectralCubicCircuitNoise
    provider = provider_class(layout=layout, spectrum=spectrum,
        port_absolute_tolerances=tolerances, check=check,
        **({"cell_for_key": cubic_cell_for_key} if cubic_cell_for_key is not None else {}))
    return provider
