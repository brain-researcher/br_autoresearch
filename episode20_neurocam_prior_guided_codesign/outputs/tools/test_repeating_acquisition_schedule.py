"""Full-scope metadata fixtures, not an adopted hardware or voltage recipe.

The fixture's4ms ADC slots, values and latency are artificial caller inputs.
Only a bounded template and selected output windows are constructed on compute;
there is no electrical propagation or104-million-record test materialization.
"""

from bisect import bisect_right
from dataclasses import replace
from itertools import chain, islice
import unittest

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import ArrayInputBinding
from daq_mux_hold import DAQGroupDrive, ExternalVoltageInput, PassiveDAQFrontend, PassiveMuxHoldGroup
from observation_nuisance import ObservationNuisanceBinding
from raw_input_hold import PhysicalAction
from repeating_acquisition_schedule import (ReturnedInputBinding, build_full_record_schedule,
                                            CYCLE_COUNT, CYCLE_SECONDS, RECORD_SECONDS)
from streaming_acquisition_readout import ReadoutWindow


REFERENCE = "artificial_test_reference"


def fixture():
    groups = []
    for index in range(24):
        sources = tuple(range(index, 64, 24))
        groups.append(PassiveMuxHoldGroup(f"adc{index}", sources, .7, 0., (.8,) * len(sources),
            (.03,) * len(sources), .05, (ExternalVoltageInput("G1", .6, .02),) if index == 0 else ()))
    frontend = PassiveDAQFrontend(tuple(groups), "additional_fixed_shunt_excludes_frontend")
    actions = []
    for tick in range(1600):
        row, slot = tick // 25, tick % 4
        selection = {}
        for index in range(24):
            source = index + 24 * slot
            selected = source if source < 64 else None
            if index == 0:
                selected = "G1" if slot == 3 else selected
                selection[f"adc{index}"] = DAQGroupDrive(selected, (("G1", float(row == 0)),), REFERENCE)
            else:
                selection[f"adc{index}"] = selected
        # Explicit caller charge is retained on the first action of each cycle.
        charge = ((4160, .001),) if tick == 0 else ()
        actions.append(PhysicalAction(tick / 400000, (tick + 1) / 400000, row, REFERENCE, selection, charge))

    transfer = QuantizerTransfer(-2., 2., (0.,), (-1., 1.))
    raw, samples, origins = [], [], []
    by_site = {site: [] for site in range(4096)}
    row_edges = tuple(row / 16000 for row in range(65))
    for period in range(400):
        for source in (*range(64), "G1"):
            adc = source % 24 if type(source) is int else 0
            slot = source // 24 if type(source) is int else 3
            tick = period * 4 + slot
            start, end = (5 * tick + 1) / 2000000, (5 * tick + 4) / 2000000
            point, complete = (2 * tick + 1) / 800000, (tick + 1) / 400000
            index = len(raw)
            raw.append(RawConversion(f"r:{index}", f"adc{adc}", f"hold{adc}",
                "signal" if type(source) is int else "marker", start, end, complete, point, transfer, True))
            samples.append(ReturnedSample(f"s:{index}", (f"r:{index}",), (1.,), 1., point, complete, None))
            origins.append(ReturnedInputBinding(f"s:{index}", source, point))
            if type(source) is int:
                by_site[(bisect_right(row_edges, point) - 1) * 64 + source].append(index)
    observations, nuisance = [], []
    for site in range(4096):
        chosen = by_site[site][2:5]  # Explicit fixture third-through-fifth arithmetic.
        if len(chosen) != 3:
            raise AssertionError("Fixture must include the registered three within-dwell source words")
        point = origins[chosen[1]].acquisition_time_s
        publication = (site // 64 + 1) / 16000 + .0001
        observations.append(Observation(f"o:{site}", f"site{site}", "pixel",
            tuple(f"s:{index}" for index in chosen), (1., 1., 1.), 3., point, publication, 16, None))
        nuisance.append(ObservationNuisanceBinding(f"o:{site}", site, point))
    # Supplied placeholder lifetime indices are NOT authority for eviction;
    # the builder derives them from all retained digital dependencies.
    readout = ReadoutWindow(0, 0., .004, tuple(raw), tuple(samples), tuple(observations),
        tuple((p.conversion_id, 0) for p in raw), tuple((p.sample_id, 0) for p in samples), tuple(nuisance))
    return dict(frontend=frontend, cycle_actions=actions, cycle_readout=readout,
        input_bindings=tuple(ArrayInputBinding(f"hold{index}", f"adc{index}", "hold", None, f"adc{index}")
                             for index in range(24)),
        returned_input_bindings=origins, marker_input_id="G1", voltage_reference_id=REFERENCE,
        max_template_actions=1600, max_template_records=60000, max_template_numeric_bytes=32 * 1024**2,
        check=lambda: None)


class RepeatingScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = fixture()

    def test_full_scope_counts_and_exact_cycle_seams_without_whole_record_expansion(self):
        schedule = build_full_record_schedule(**self.parameters)
        self.assertEqual(schedule.acquisition_window, (0., 16.))
        self.assertEqual(schedule.metadata["physical_ADC_groups"], 24)
        self.assertEqual(schedule.metadata["neural_source_lines"], 64)
        self.assertEqual(schedule.metadata["full_returned_source_words"], 102400000)
        self.assertEqual(schedule.metadata["full_G1_returned_words"], 1600000)
        self.assertEqual(schedule.metadata["full_physical_conversions"], 104000000)
        self.assertFalse(schedule.metadata["source_stream_proxy_is_physical_ADC_clock"])
        self.assertEqual(schedule.metadata["full_pixel_observations"], 16384000)
        self.assertEqual(schedule.metadata["retained_template_actions"], 1600)
        self.assertEqual(schedule.metadata["full_physical_actions"], 6400000)
        first = schedule.readout_window(0)
        following = schedule.readout_window(1)
        self.assertEqual(first.end, following.start)
        self.assertEqual(first.conversions[-1].completed_at, following.start)
        self.assertEqual((following.conversions[0].conversion_id, following.returned_samples[0].sample_id,
                          following.observations[0].observation_id), ("r:26000", "s:26000", "o:4096"))
        self.assertEqual(following.returned_samples[0].conversion_ids, ("r:26000",))
        self.assertEqual(following.observations[0].weights, (1., 1., 1.))
        self.assertIsNone(following.observations[0].overhead_bits)
        final = schedule.readout_window(CYCLE_COUNT - 1)
        self.assertEqual(final.end, RECORD_SECONDS)
        self.assertEqual(final.conversions[-1].completed_at, RECORD_SECONDS)
        self.assertEqual(final.conversions[-1].conversion_id, "r:103999999")
        self.assertTrue(all(binding.acquisition_time_s < RECORD_SECONDS for binding in final.nuisance_bindings))
        cycle0 = tuple(schedule.physical_actions_for_cycle(0))
        cycle1 = tuple(schedule.physical_actions_for_cycle(1))
        self.assertEqual(cycle0[-1].end, cycle1[0].start)
        self.assertEqual(cycle1[0].injected_charge_coulomb, ((4160, .001),))
        self.assertTrue(all(not action.injected_charge_coulomb for action in cycle1[1:]))
        self.assertEqual(tuple(islice(schedule.physical_action_factory(), 2)), cycle0[:2])
        self.assertEqual(tuple(schedule.physical_actions_for_cycle(3999))[-1].end, RECORD_SECONDS)

    def test_delayed_publication_keeps_dependencies_and_never_clamps_timestamps(self):
        schedule = build_full_record_schedule(**self.parameters)
        first = schedule.readout_window(0)
        delayed = first.observations[-1]
        self.assertGreater(delayed.available_at, first.end)
        sample_last, raw_last = dict(first.sample_last_use_windows), dict(first.raw_last_use_windows)
        for sample_id in delayed.sample_ids:
            self.assertEqual(sample_last[sample_id], 1)
            sample = first.returned_samples[int(sample_id.split(":")[1])]
            for raw_id in sample.conversion_ids:
                self.assertEqual(raw_last[raw_id], 1)
        final = schedule.readout_window(3999)
        self.assertGreater(final.observations[-1].available_at, 16.)
        self.assertEqual(schedule.final_publication_horizon, final.observations[-1].available_at)
        self.assertEqual(schedule.final_publication_horizon, 3999 * CYCLE_SECONDS + .0041)
        self.assertEqual(dict(final.sample_last_use_windows)[final.observations[-1].sample_ids[0]], 3999)
        self.assertEqual(final.nuisance_bindings[-1].physical_site_index, 4095)
        self.assertEqual(final.nuisance_bindings[-1].acquisition_time_s,
                         3999 * CYCLE_SECONDS + self.parameters["cycle_readout"].nuisance_bindings[-1].acquisition_time_s)

    def test_incomplete_cross_cycle_and_changed_source_bindings_refuse_and_template_is_owned(self):
        parameters = {**self.parameters, "cycle_actions": list(self.parameters["cycle_actions"]),
                      "returned_input_bindings": list(self.parameters["returned_input_bindings"])}
        schedule = build_full_record_schedule(**parameters)
        parameters["cycle_actions"].clear()
        parameters["returned_input_bindings"].clear()
        self.assertEqual(len(tuple(schedule.physical_actions_for_cycle(0))), 1600)
        self.assertEqual(len(schedule.returned_input_bindings), 26000)
        original = self.parameters["cycle_readout"]
        changed = replace(original.observations[0], sample_ids=("s:26000",), weights=(1.,), divisor=1.)
        bad_template = replace(original, observations=(changed, *original.observations[1:]))
        with self.assertRaisesRegex(ValueError, "Cross-cycle"):
            build_full_record_schedule(**{**self.parameters, "cycle_readout": bad_template})
        origins = list(self.parameters["returned_input_bindings"])
        # Source0 and24 share a physical ADC in this explicit fixture: the
        # mismatch must still fail the complete64-source proxy counts.
        origins[0] = replace(origins[0], input_label=24)
        with self.assertRaisesRegex(ValueError, "400 explicit returned words"):
            build_full_record_schedule(**{**self.parameters, "returned_input_bindings": origins})
        bad_actions = list(self.parameters["cycle_actions"])
        bad_actions[-1] = replace(bad_actions[-1], gate_row=0)
        with self.assertRaisesRegex(ValueError, "all64 ascending"):
            build_full_record_schedule(**{**self.parameters, "cycle_actions": bad_actions})
        with self.assertRaisesRegex(ValueError, "numeric-byte"):
            build_full_record_schedule(**{**self.parameters, "max_template_numeric_bytes": 1})
        extra_bindings = chain(self.parameters["input_bindings"], self.parameters["input_bindings"][:1])
        with self.assertRaisesRegex(ValueError, "Input bindings exceed"):
            build_full_record_schedule(**{**self.parameters, "input_bindings": extra_bindings})


if __name__ == "__main__":
    unittest.main()
