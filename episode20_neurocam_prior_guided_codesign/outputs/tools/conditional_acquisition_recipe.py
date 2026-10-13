"""One selected conditional G0/P-full DAQ recipe, never authenticated wiring.

All 24 physical ADCs and 64 neural sources remain present for 16 seconds.
The four-slot timing, uniform quantizer, external G1 waveform, switch charge,
hold parameters and digital operator below are prospective engineering choices.
They do not identify NeuroCam/NI settings, establish calibration or certify
power/bitrate parity. Physical states persist; only control metadata repeats.
No source files, saved worlds or empirical records are opened by this module.
"""

from bisect import bisect_right
import math
from numbers import Real
from types import MappingProxyType

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import ArrayInputBinding
from daq_mux_hold import DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend, PassiveMuxHoldGroup
from raw_input_hold import PhysicalAction
from repeating_acquisition_schedule import ReturnedInputBinding, build_full_record_schedule, _shift
from streaming_acquisition_readout import ReadoutWindow


REFERENCE_ID = "conditional_R3_zero_reference"
MARKER_ID = "G1"
ADC_IDS = tuple(f"conditional_adc_{index:02d}" for index in range(24))
HOLD_IDS = tuple(f"daq_hold_{adc_id}" for adc_id in ADC_IDS)
INPUT_IDS = tuple(f"hold_input_{index:02d}" for index in range(24))
CYCLE_COUNT = 4000
CYCLE_SECONDS = 0.004
RECORD_SECONDS = 16.0
HALF_MICROSECOND_TICKS_PER_SECOND = 2_000_000
SLOT_TICKS = 5
ROW_TICKS = 125
CYCLE_TICKS = 8000
SLOTS_PER_CYCLE = 1600
QUANTIZER_BITS = 18
QUANTIZER_LIMIT_VOLTS = 0.010
HOLD_CAPACITANCE_FARAD = 20e-12
TRACK_CONDUCTANCE_SIEMENS = 1e-3
OFF_CONDUCTANCE_SIEMENS = 1e-10
HOLD_LEAK_CONDUCTANCE_SIEMENS = 1e-10
ROW_SOURCE_CHARGE_COULOMB = 3.5e-17
MUX_HOLD_CHARGE_COULOMB = 2e-17
MARKER_HIGH_VOLTS = 0.005
TEMPLATE_ACTION_LIMIT = 1600
TEMPLATE_RECORD_LIMIT = 56_496
TEMPLATE_NUMERIC_BYTES_LIMIT = 16 * 1024**2


def _time(ticks):
    """Round one integer clock coordinate once, including every cycle seam."""
    return ticks / HALF_MICROSECOND_TICKS_PER_SECOND


def conditional_noise_cell(key):
    """Fixed original mux slot, with the exact schedule endpoint arithmetic.

    Field refinement, ADC apertures and storage splits keep this same origin.
    The bounded neighbor check handles quotient roundoff near a clock edge;
    a key on an actual edge refuses instead of assigning an ambiguous cell.
    """
    if isinstance(key, bool) or not isinstance(key, Real) or not math.isfinite(key) or not 0 < key < RECORD_SECONDS:
        raise ValueError("A finite midpoint inside the complete conditional record is required")
    cycle_guess = int(key / CYCLE_SECONDS)
    for cycle in range(max(0, cycle_guess - 1), min(CYCLE_COUNT, cycle_guess + 2)):
        slot_guess = int((key - cycle * CYCLE_SECONDS) / _time(SLOT_TICKS))
        for slot in range(max(0, slot_guess - 1), min(SLOTS_PER_CYCLE, slot_guess + 2)):
            start = _shift(cycle, _time(slot * SLOT_TICKS))
            end = _shift(cycle, _time((slot + 1) * SLOT_TICKS))
            if start < key < end:
                return start, end, start + (end - start) / 2
    raise ValueError("Noise query is not strictly inside an original physical slot")


def build_conditional_frontend():
    """The selected 3-card/24-ADC mathematical assignment and finite loading.

    ADC j is board j//8, group j%8. Input slot k is that board's
    ai[j%8 + 8*k]. Neural source j+24*k occupies its declared slot when <64;
    G1 uses board0/ai24, the fourth input of ADC0. This is a caller-selected
    assignment within the fixed family, not evidence of historical wiring.
    """
    groups = []
    for index, adc_id in enumerate(ADC_IDS):
        sources = tuple(range(index, 64, 24))
        groups.append(PassiveMuxHoldGroup(adc_id, sources,
            HOLD_CAPACITANCE_FARAD, 0.0,
            (TRACK_CONDUCTANCE_SIEMENS,) * len(sources),
            (OFF_CONDUCTANCE_SIEMENS,) * len(sources),
            HOLD_LEAK_CONDUCTANCE_SIEMENS,
            (ExternalVoltageInput(MARKER_ID, TRACK_CONDUCTANCE_SIEMENS,
                                  OFF_CONDUCTANCE_SIEMENS),) if index == 0 else ()))
    return PassiveDAQFrontend(tuple(groups), "additional_fixed_shunt_excludes_frontend")


def _quantizer(check):
    # 2**18 equal bins over [-10,+10]mV; representatives are bin centres.
    # Endpoint inputs are in-range, threshold ties choose the upper bin, and
    # outside-range inputs saturate with the existing distinct clipping flag.
    count = 1 << QUANTIZER_BITS
    step = 2 * QUANTIZER_LIMIT_VOLTS / count
    levels, thresholds = [], []
    for index in range(count):
        if index % 4096 == 0:
            check()
        levels.append(-QUANTIZER_LIMIT_VOLTS + (index + 0.5) * step)
        if index + 1 < count:
            thresholds.append(-QUANTIZER_LIMIT_VOLTS + (index + 1) * step)
    return QuantizerTransfer(-QUANTIZER_LIMIT_VOLTS, QUANTIZER_LIMIT_VOLTS,
                             tuple(thresholds), tuple(levels))


def build_conditional_schedule(frontend, *, state_ids, check):
    """Bind the finite cycle to the supplied full-array state basis.

    Only one cycle's records and one shared quantizer are constructed. Existing
    repeating factories produce every cycle, with exact whole-plan preflight
    still owned by the streaming executor. This function performs no physics.
    The source/hold indices must be those of the assembled selected circuit;
    no detached diagnostic source is substituted for a physical ADC hold.
    """
    if not callable(check):
        raise ValueError("An explicit construction/deadline check is required")
    check()
    if frontend != build_conditional_frontend():
        raise ValueError("This fixed conditional recipe requires its selected complete frontend")
    if (not isinstance(state_ids, tuple) or any(not isinstance(name, str) for name in state_ids)
            or len(set(state_ids)) != len(state_ids)
            or state_ids[4096:4160] != tuple(f"source_{source:02d}" for source in range(64))
            or state_ids[-24:] != HOLD_IDS):
        raise ValueError("Supply the actual unique full-array source and final24 hold state IDs")
    source_nodes = tuple(range(4096, 4160))
    hold_nodes = tuple(range(len(state_ids) - 24, len(state_ids)))
    actions = []
    # Every start-of-record selected switch changes from the declared all-off
    # initialization. The final cycle slot also leaves each next first-slot
    # switch different, so the same explicit charge sequence repeats correctly.
    previous = (None,) * 24
    for tick in range(SLOTS_PER_CYCLE):
        check()
        row, slot = tick // 25, tick % 4
        selected, selection = [], {}
        for index, adc_id in enumerate(ADC_IDS):
            source = index + 24 * slot
            member = source if source < 64 else None
            if index == 0:
                member = MARKER_ID if slot == 3 else member
                selection[adc_id] = DAQGroupDrive(member,
                    ((MARKER_ID, MARKER_HIGH_VOLTS if row == 0 else 0.0),), REFERENCE_ID)
            else:
                selection[adc_id] = member
            selected.append(member)
        charge = []
        if tick % 25 == 0:
            charge.extend((node, ROW_SOURCE_CHARGE_COULOMB * (1 if source % 2 == 0 else -1))
                          for source, node in enumerate(source_nodes))
        charge.extend((hold_nodes[index], MUX_HOLD_CHARGE_COULOMB * (1 if slot % 2 == 0 else -1))
                      for index, member in enumerate(selected) if member != previous[index])
        actions.append(PhysicalAction(_time(SLOT_TICKS * tick), _time(SLOT_TICKS * (tick + 1)),
            row, REFERENCE_ID, selection, tuple(charge)))
        previous = tuple(selected)

    quantizer = _quantizer(check)
    raw, returned, origins = [], [], []
    by_site = [[] for _ in range(4096)]
    marker_samples = []
    row_edges = tuple(_time(row * ROW_TICKS) for row in range(65))
    for period in range(400):
        for label in (*range(64), MARKER_ID):
            check()
            adc, slot = (label % 24, label // 24) if type(label) is int else (0, 3)
            tick = 4 * period + slot
            start_ticks = SLOT_TICKS * tick
            aperture_start, aperture_end = _time(start_ticks + 1), _time(start_ticks + 4)
            completed_at = _time(start_ticks + SLOT_TICKS)
            # Mid-aperture is a quarter-microsecond clock coordinate, rounded once.
            point = (2 * start_ticks + 5) / (2 * HALF_MICROSECOND_TICKS_PER_SECOND)
            index = len(raw)
            raw.append(RawConversion(f"r:{index}", ADC_IDS[adc], INPUT_IDS[adc],
                "signal" if type(label) is int else "marker", aperture_start, aperture_end,
                completed_at, point, quantizer, True))
            returned.append(ReturnedSample(f"s:{index}", (f"r:{index}",), (1.0,), 1.0,
                                           point, completed_at, None))
            origins.append(ReturnedInputBinding(f"s:{index}", label, point))
            if type(label) is int:
                row = bisect_right(row_edges, point) - 1
                by_site[64 * row + label].append(index)
            else:
                marker_samples.append(index)

    observations = []
    for site, candidates in enumerate(by_site):
        check()
        if len(candidates) not in (6, 7):
            raise ValueError("Every source/row must retain its six or seven actual returned words")
        chosen = candidates[2:5]
        observations.append(Observation(f"o:{site}", f"site{site}", "pixel",
            tuple(f"s:{index}" for index in chosen), (1.0, 1.0, 1.0), 3.0,
            origins[chosen[1]].acquisition_time_s, row_edges[site // 64 + 1], 32, None))
    for index in marker_samples:
        check()
        word = returned[index]
        observations.append(Observation(f"o:{len(observations)}", MARKER_ID, "marker",
            (word.sample_id,), (1.0,), 1.0, word.assigned_timestamp, word.available_at, 32, None))
    # No N-family overlay or additional digital filter is selected for B2.
    # Published filter settings are unresolved, not silently reconstructed.
    readout = ReadoutWindow(0, 0.0, CYCLE_SECONDS, tuple(raw), tuple(returned), tuple(observations),
        tuple((plan.conversion_id, 0) for plan in raw),
        tuple((plan.sample_id, 0) for plan in returned))
    return build_full_record_schedule(frontend=frontend, cycle_actions=actions, cycle_readout=readout,
        input_bindings=tuple(ArrayInputBinding(INPUT_IDS[index], adc_id, "hold", None, adc_id)
                             for index, adc_id in enumerate(ADC_IDS)),
        returned_input_bindings=origins, marker_input_id=MARKER_ID, voltage_reference_id=REFERENCE_ID,
        max_template_actions=TEMPLATE_ACTION_LIMIT, max_template_records=TEMPLATE_RECORD_LIMIT,
        max_template_numeric_bytes=TEMPLATE_NUMERIC_BYTES_LIMIT, check=check)


def acquisition_recipe_metadata():
    """Algebraic counts/assumptions; no completed-run or measured-cost claim."""
    return MappingProxyType(dict(
        scope="one_conditional_full16s_B2_seed2001_G0_cost_measurement",
        authenticated_DAQ_configuration=False, calibrated_reference=False, candidate_scored=False,
        reference_id=REFERENCE_ID, record_seconds=RECORD_SECONDS, cycles=CYCLE_COUNT,
        rows=64, source_lines=64, physical_sites=4096, physical_ADC_groups=24, switch_paths=65,
        source_ADC_assignment=tuple((source, ADC_IDS[source % 24], source % 24 // 8,
                                    source % 8 + 8 * (source // 24)) for source in range(64)),
        marker_ADC_assignment=(MARKER_ID, ADC_IDS[0], 0, 24),
        cycle_seconds=CYCLE_SECONDS, row_dwell_seconds=62.5e-6, mux_slot_seconds=2.5e-6,
        raw_aperture_offset_seconds=0.5e-6, raw_aperture_seconds=1.5e-6,
        conversion_completion_offset_seconds=2.5e-6,
        raw_quantizer_bits=QUANTIZER_BITS, raw_quantizer_range_volts=(-0.010, 0.010),
        raw_quantizer_representatives="equal_bin_centres_threshold_tie_upper",
        returned_operator="one_decoded_raw_conversion_identity_no_requantization",
        pixel_operator="third_through_fifth_source_words_inside_actual_row_mean",
        pixel_assigned_time="fourth_word_aperture_midpoint", pixel_availability="row_end",
        marker_operator="every_G1_returned_word_identity_at_conversion_completion",
        further_digital_filter="explicit_identity_not_recovered_published_filter",
        observation_payload_bits=32, observation_packet_overhead_bits=None,
        pixel_payload_convention="uint32_sum_of_three_18bit_raw_codes_known_divisor3_and_bin_centre_decode",
        marker_payload_convention="18bit_raw_code_in_uint32_no_extra_output_requantization",
        payload_format_is_conditional_not_authenticated=True,
        marker_levels_volts=(0.0, MARKER_HIGH_VOLTS), marker_high_condition="gate_row_zero",
        hold_capacitance_farad=HOLD_CAPACITANCE_FARAD, hold_initial_volts=0.0,
        track_conductance_siemens=TRACK_CONDUCTANCE_SIEMENS,
        off_conductance_siemens=OFF_CONDUCTANCE_SIEMENS,
        hold_leak_conductance_siemens=HOLD_LEAK_CONDUCTANCE_SIEMENS,
        row_source_charge_coulomb=ROW_SOURCE_CHARGE_COULOMB,
        row_source_charge_sign="positive_even_source_negative_odd_source",
        mux_hold_charge_coulomb=MUX_HOLD_CHARGE_COULOMB,
        mux_hold_charge_sign="positive_even_slot_negative_odd_slot_only_selection_changes",
        startup="zero_stored_states_all_switches_off_then_first_action_charge_startup_retained",
        row_charge_entries_per_cycle=4096, mux_charge_entries_per_cycle=35200,
        template_actions=1600, template_raw_conversions=26000, template_returned_words=26000,
        template_pixel_observations=4096, template_marker_observations=400,
        source_returned_words_per_cycle=400, source_returned_words_per_second=100000,
        full_physical_actions=6_400_000, full_neural_returned_words=102_400_000,
        full_marker_raw_and_returned_words=1_600_000,
        full_raw_conversions=104_000_000, full_returned_words=104_000_000,
        full_pixel_observations=16_384_000, full_marker_observations=1_600_000,
        full_observations=17_984_000, full_aperture_support_records=50_752_000,
        full_unrefined_electrical_intervals=19_200_000,
        aggregate_selected_raw_conversions_per_second=6_500_000,
        published_source_stream_proxy_per_second=6_400_000,
        source_stream_proxy_is_physical_ADC_clock=False,
        selected_raw_conversion_rates_by_ADC=(400000, *([300000] * 15), *([200000] * 8)),
        selected_raw_word_payload_bits=1_872_000_000,
        selected_observation_payload_bits=575_488_000,
        raw_conversions_and_returned_words_are_separate_counts=True,
        marker_and_startup_work_included=True,
        hardware_averaging_auto_zero_chopping="not_enabled_in_this_conditional_recipe_not_verified_absent",
        writer_identifier_bytes=64, writer_raw_record_bytes=90, writer_observation_record_bytes=235,
        writer_aperture_support_record_bytes=80, writer_interval_record_bytes=25,
        fixed_raw_observation_support_payload_bytes=17_646_400_000,
        unrefined_interval_payload_bytes=480_000_000,
        baseline_table_payload_bytes=18_126_400_000,
        interval_cap=40_000_000, table_payload_at_interval_cap_bytes=18_646_400_000,
        payload_exclusions="state_arrays_settings_terminal_JSON_HDF5_chunks_metadata_and_filesystem_overhead",
        complete_power_or_bitrate_parity_available=False,
        full_record_completion_is_not_predicted=True))
