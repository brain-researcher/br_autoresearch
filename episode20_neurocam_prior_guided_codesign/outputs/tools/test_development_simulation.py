"""Four focused orchestration checks; private fixtures, no retained world reads.

The arbitrary circuit constants below test state/charge/readout integration;
they are never device estimates or proposed EP20 run parameters.
"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import AcquisitionLimits, ArrayAcquisition, ArrayInputBinding
from cpe_array import AssemblyLimits, NeurocamCPEArray
from development_worlds import _Grid, _build, _save_world, DevelopmentWorld
from development_simulation import (load_saved_development_world, run_development_simulation,
                                    save_simulation_result, _AccountingArray)
from raw_input_hold import PhysicalAction, compile_midpoint_hold
from raw_world_time import RawWorldTime
from surface_to_contact import registered_pad_geometry, build_contact_factors
from test_cpe_array import ideal_contact, fractional_contact
from test_daq_mux_hold import frontend, group
from test_sparse_array_rc import controls

LIMIT = 20_000_000


def parameters():
    contacts = [ideal_contact()] * 4096
    contacts[0] = fractional_contact()
    access = np.zeros((64, 64))
    access[0, 1] = .3
    return dict(contacts=contacts, access_resistance_ohm=access, access_resistance_owner="assembler",
        voltage_reference_id="orchestration_fixture_reference", initial_field_volts=np.zeros((64, 64)),
        initial_pad_volts=np.zeros((64, 64)), initial_source_volts=np.full(64, .01),
        initial_contact_state_volts=np.array([.002, -.001, .005]),
        pad_capacitance_farad=np.ones((64, 64)), source_capacitance_farad=np.ones(64),
        mutual_capacitance_edges=(), tft_on_conductance_siemens=np.full((64, 64), .2),
        tft_off_conductance_siemens=np.full((64, 64), .1),
        pad_leak_conductance_siemens=np.zeros((64, 64)), source_load_conductance_siemens=np.full(64, .03),
        other_resistor_edges=(), limits=AssemblyLimits(5000, 10000, 10000, 32 * 1024 * 1024),
        daq_frontend=frontend(group()))


def arguments(world):
    reference = "orchestration_fixture_reference"
    transfer = QuantizerTransfer(-1., 1., (0.,), (-.5, .5))
    raw = [RawConversion("raw_a", "fixture_adc_a", "hold_input", "signal", .002, .006, .006, .002, transfer, True),
           RawConversion("raw_b", "fixture_adc_a", "hold_input", "signal", .012, .018, .018, .012, transfer, True)]
    returned = [ReturnedSample("word_" + r.conversion_id, (r.conversion_id,), (1.,), 1.,
                               r.assigned_timestamp, r.completed_at, None) for r in raw]
    observations = [Observation("obs_" + r.conversion_id, "fixture_site", "pixel", (s.sample_id,),
                                (1.,), 1., r.assigned_timestamp, .03, 8, None)
                    for r, s in zip(raw, returned)]
    return dict(world=world, geometry_id="G0", short_axis="x", circuit_parameters=parameters(),
        physical_actions=[PhysicalAction(0., .01, 0, reference, {"fixture_adc_a": 0}, ((4163, .02),)),
                          PhysicalAction(.01, .02, 1, reference, {"fixture_adc_a": 1}, ((4096, .01),))],
        input_bindings=[ArrayInputBinding("hold_input", "fixture_adc_a", "hold", None, "fixture_adc_a")],
        conversions=raw, returned_samples=returned, observations=observations,
        acquisition_window=(0., .02), final_publication_horizon=.03, solver_controls=controls(),
        acquisition_limits=AcquisitionLimits(100, 100, 10, 10, 10, 1024 * 1024),
        epsilon_input_volts=1e-4, max_hold_work=1000, max_elements=LIMIT,
        max_total_operator_products=200000, max_seconds=60.,
        settings_reference="private_test_fixture_not_a_device_recipe")


class DevelopmentSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = _build("W1", 2001, _Grid(), check=lambda: None, max_elements=LIMIT, include_scales=False)

    def test_saved_world_exact_group_roundtrip_limits_and_no_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worlds.h5"
            with h5py.File(path, "x") as container:
                _save_world(container, self.world, lambda: None)
                container["closed_unrelated_world"] = h5py.ExternalLink("must_not_be_opened.h5", "/")
            kwargs = dict(condition_id="W1", replicate_seed=2001, max_elements=LIMIT,
                          max_array_bytes=16 * 1024 * 1024, max_metadata_bytes=1024 * 1024)
            with patch("development_worlds.build_development_world", side_effect=AssertionError("no regeneration")):
                loaded = load_saved_development_world(path, **kwargs)
            self.assertEqual(loaded.metadata, self.world.metadata)
            for key in ("spatial_modes", "raw_drivers", "primary_band_drivers"):
                np.testing.assert_array_equal(getattr(loaded, key), getattr(self.world, key))
                self.assertFalse(getattr(loaded, key).flags.writeable)
            with self.assertRaises(ValueError):
                load_saved_development_world(path, **{**kwargs, "max_array_bytes": 1})
            with self.assertRaises(ValueError):
                load_saved_development_world(path, **{**kwargs, "max_metadata_bytes": 1})
            with h5py.File(path, "r+") as container:
                container["W1_seed_2001"].attrs["complete"] = False
            with self.assertRaises(ValueError):
                load_saved_development_world(path, **kwargs)

    def test_full_state_matches_direct_components_and_retains_work_readout_output(self):
        args = arguments(self.world)
        result = run_development_simulation(**args)
        self.assertEqual(result["status"], "complete", result.get("failure"))
        geometry = registered_pad_geometry("G0", short_axis="x", max_elements=LIMIT)
        contacts = build_contact_factors(self.world, geometry, max_elements=LIMIT)
        raw_time = RawWorldTime(self.world, max_elements=LIMIT)
        edges = sorted({t for r in args["conversions"] for t in (r.aperture_start, r.aperture_end)})
        hold = compile_midpoint_hold(raw_time, contacts, args["physical_actions"], epsilon_input_volts=1e-4,
                                     max_intervals=100, max_work=1000, additional_boundaries_s=edges)
        array = NeurocamCPEArray(**args["circuit_parameters"])
        direct = ArrayAcquisition(array=array, physical_intervals=hold.intervals, field_provider=hold.field_provider,
            **{key: args[key] for key in ("input_bindings", "conversions", "returned_samples", "observations",
                                         "acquisition_window", "final_publication_horizon")},
            controls=args["solver_controls"], limits=args["acquisition_limits"])
        records, original = [], array.advance_row
        def record(**kwargs):
            value = original(**kwargs)
            records.append(value)
            return value
        with patch.object(array, "advance_row", side_effect=record):
            expected = direct.run()
        self.assertEqual(result["final_state_volts"], array.volts)
        self.assertEqual(result["final_field_volts"], array.previous_field_volts)
        self.assertEqual(len(result["state_ids"]), 4164)
        self.assertEqual(result["readout"]["raw_values"], expected["raw_values"])
        self.assertEqual(result["readout"]["observations"], expected["observations"])
        self.assertEqual(result["readout"]["actual_internal_conversions"], 2)
        self.assertEqual(result["readout"]["propagation_operator_products"],
                         result["accounting"]["confirmed_operator_products"])
        for key in ("field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
                    "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
                    "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
                    "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule"):
            self.assertAlmostEqual(result["accounting"][key], sum(getattr(r, key) for r in records), places=14)
        self.assertEqual(result["accounting"]["physical_initial_energy_joule"], records[0].physical_energy_before_boundary_joule)
        self.assertEqual(result["accounting"]["physical_final_energy_joule"], records[-1].physical_energy_end_joule)
        self.assertIsNone(result["readout"]["modeled_power_watt"])
        self.assertFalse(result["covers_full_16_second_record"])
        self.assertFalse(result["actual_DAQ_configuration_qualified"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "result.h5"
            save_simulation_result(path, result, max_payload_bytes=1024 * 1024)
            with h5py.File(path, "r") as container:
                self.assertTrue(container.attrs["record_complete"])
                np.testing.assert_array_equal(container["final_state_volts"][...], array.volts)
                self.assertEqual(len(container["raw_values_json"]), 2)
                metadata = json.loads(container["metadata_json"].asstr()[()])
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["readout"]["actual_internal_conversions"], 2)
            with self.assertRaises(FileExistsError):
                save_simulation_result(path, result, max_payload_bytes=1024 * 1024)

    def test_operator_and_elapsed_caps_preserve_partial_state_without_replay(self):
        args = arguments(self.world)
        limited = run_development_simulation(**{**args, "max_total_operator_products": 1})
        self.assertEqual(limited["status"], "failed")
        self.assertEqual(limited["accounting"]["failed_call_operator_upper_bound"], 1)
        self.assertIsNone(limited["accounting"]["failed_call_operator_products"])
        self.assertGreater(limited["accounting"]["measured_failed_solver_seconds"], 0.)
        self.assertEqual(limited["accounting"]["measured_solver_seconds"],
                         limited["accounting"]["measured_failed_solver_seconds"])
        self.assertEqual(limited["final_state_volts"], limited["initial_state_volts"])
        self.assertIsNone(limited["readout"])
        clock, calls = [0.], []
        original = _AccountingArray.advance_row
        def expire_after_one(array, **kwargs):
            value = original(array, **kwargs)
            calls.append(value)
            clock[0] = 2.
            return value
        with patch("development_simulation.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(_AccountingArray, "advance_row", new=expire_after_one):
            stopped = run_development_simulation(**{**args, "max_seconds": 1.})
        self.assertEqual(stopped["status"], "failed")
        self.assertEqual(len(calls), 1)
        self.assertEqual(stopped["accounting"]["returned_electrical_intervals"], 1)
        self.assertEqual(stopped["confirmed_physical_time"], .002)
        self.assertNotEqual(stopped["final_state_volts"], stopped["initial_state_volts"])
        self.assertIsNone(stopped["readout"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.h5"
            save_simulation_result(path, stopped, max_payload_bytes=1024 * 1024)
            with h5py.File(path, "r") as container:
                self.assertTrue(container.attrs["record_complete"])  # Complete record of a failed attempt.
                metadata = json.loads(container["metadata_json"].asstr()[()])
                self.assertEqual(metadata["status"], "failed")

    def test_nuisance_and_access_ownership_are_not_silently_reinterpreted(self):
        metadata = dict(self.world.metadata, condition_id="N1", world_id="N1:2001")
        nuisance = DevelopmentWorld(self.world.spatial_modes, self.world.raw_drivers,
                                    self.world.primary_band_drivers, metadata, {})
        with patch("development_simulation._AccountingArray", side_effect=AssertionError("must not construct")):
            unresolved = run_development_simulation(**arguments(nuisance))
            self.assertEqual(unresolved["status"], "unresolved")
            self.assertFalse(unresolved["electrical_state_advanced"])
            args = arguments(self.world)
            args["circuit_parameters"]["access_resistance_owner"] = "upstream"
            with self.assertRaises(ValueError):
                run_development_simulation(**args)
            args["circuit_parameters"]["access_resistance_owner"] = "assembler"
            del args["circuit_parameters"]["daq_frontend"]
            with self.assertRaises(ValueError):
                run_development_simulation(**args)


if __name__ == "__main__":
    unittest.main()
