"""Full-chain private fixtures; constants are not proposed device recipes."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from array_acquisition import ArrayAcquisition
from cpe_array import NeurocamCPEArray
from development_worlds import _Grid, _build
from raw_input_hold import compile_midpoint_hold
from raw_world_time import RawWorldTime
from surface_to_contact import registered_pad_geometry, build_contact_factors
from streaming_acquisition_readout import ReadoutWindow, StreamingReadoutLimits
from streaming_array_acquisition import StreamingArrayLimits
from streaming_array_acquisition import StreamingArrayAcquisition
from streaming_development_simulation import run_streaming_development_simulation
from streaming_input_plan import bind_hold_windows
from test_development_simulation import arguments, LIMIT
from observation_nuisance import ObservationNuisanceBinding
from circuit_noise_provider import SpectralCircuitNoise
from physical_noise_ports import NoiseLimits, NodalCurrentPort
from spectral_noise import LatentQuadrature, SpectralNoise, SpectralNoiseLimits


def cases(world, path):
    eager = arguments(world)
    raws = [replace(value, conversion_id=f"r:{index}") for index, value in enumerate(eager["conversions"])]
    samples = [replace(value, sample_id=f"s:{index}", conversion_ids=(raws[index].conversion_id,))
               for index, value in enumerate(eager["returned_samples"])]
    observations = [replace(value, observation_id=f"o:{index}", sample_ids=(samples[index].sample_id,))
                    for index, value in enumerate(eager["observations"])]
    def windows():
        yield ReadoutWindow(0, 0., .004, (raws[0],), (samples[0],), (observations[0],),
                            (("r:0", 1),), (("s:0", 1),))
        yield ReadoutWindow(1, .004, .02, (raws[1],), (samples[1],), (observations[1],),
                            (("r:1", 1),), (("s:1", 1),))
    streaming = {key: eager[key] for key in ("world", "geometry_id", "short_axis", "circuit_parameters",
        "physical_actions", "input_bindings", "acquisition_window", "final_publication_horizon",
        "solver_controls", "epsilon_input_volts", "max_hold_work", "max_elements", "max_seconds", "settings_reference")}
    streaming.update(readout_window_factory=windows, max_hold_intervals=100,
        array_limits=StreamingArrayLimits(3, 1, 100, 200, 100, 300, 200, 16 * 1024 * 1024, 96, 200000),
        readout_limits=StreamingReadoutLimits(10, 10, 10, 300, 3, 10, 10, 10, 1024 * 1024, 10),
        result_path=path, output_limits=dict(max_payload_bytes=8 * 1024 * 1024,
            max_records=1000, buffer_records=8, identifier_bytes=128, max_dependencies_per_observation=2))
    return eager, streaming, raws, samples, observations


class StreamingDevelopmentSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = _build("W1", 2001, _Grid(), check=lambda: None, max_elements=LIMIT, include_scales=False)

    def test_complete_chain_matches_same_held_input_eager_execution_across_seam(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.h5"
            eager, kwargs, raws, samples, observations = cases(self.world, path)
            phases, callbacks = [], []
            result = run_streaming_development_simulation(**kwargs,
                execution_check=lambda: callbacks.append(True), on_phase=phases.append)
            self.assertEqual(result["status"], "complete", result)
            self.assertTrue(callbacks)
            self.assertEqual(phases, ["surface and hold preparation", "electrical assembly",
                "exclusive output initialization", "whole-plan preflight",
                "persistent electrical/readout execution"])
            replay_path = Path(directory) / "replayed.h5"
            replay_args = dict(kwargs, physical_actions=lambda: iter(eager["physical_actions"]),
                               result_path=replay_path)
            replayed = run_streaming_development_simulation(**replay_args)
            self.assertEqual(replayed["status"], "complete", replayed)
            self.assertTrue(replayed["physical_action_factory_replayed"])
            self.assertFalse(replayed["compact_full_hold_plan_retained"])
            with h5py.File(path, "r") as retained, h5py.File(replay_path, "r") as replay:
                for name in ("final_state_volts", "final_field_volts", "raw_values",
                             "observations", "interval_evidence", "aperture_support"):
                    np.testing.assert_array_equal(retained[name][...], replay[name][...])
            self.assertEqual(result["readout"]["actual_internal_conversions"], 2)
            self.assertFalse(result["covers_full_16_second_record"])
            self.assertFalse(result["candidate_scored"])
            geometry = registered_pad_geometry("G0", short_axis="x", max_elements=LIMIT)
            contact = build_contact_factors(self.world, geometry, max_elements=LIMIT)
            hold = compile_midpoint_hold(RawWorldTime(self.world, max_elements=LIMIT), contact,
                eager["physical_actions"], epsilon_input_volts=kwargs["epsilon_input_volts"],
                max_intervals=100, max_work=1000)
            factory = bind_hold_windows(hold, kwargs["readout_window_factory"],
                                        max_windows=3, max_window_fragments=100, check=lambda: None)
            fragments = [fragment.interval for window in factory() for fragment in window.physical_intervals]
            array = NeurocamCPEArray(**eager["circuit_parameters"])
            direct = ArrayAcquisition(array=array, physical_intervals=fragments,
                field_provider=hold.field_provider, conversions=raws, returned_samples=samples,
                observations=observations, input_bindings=eager["input_bindings"],
                acquisition_window=eager["acquisition_window"],
                final_publication_horizon=eager["final_publication_horizon"],
                controls=eager["solver_controls"], limits=eager["acquisition_limits"]).run()
            with h5py.File(path, "r") as saved:
                self.assertTrue(saved.attrs["record_complete"])
                self.assertEqual(saved.attrs["simulation_status"], "complete")
                np.testing.assert_allclose(saved["final_state_volts"][...], array.volts, rtol=1e-10, atol=1e-12)
                np.testing.assert_array_equal(saved["final_field_volts"][...], array.previous_field_volts)
                np.testing.assert_array_equal(saved["raw_values"]["code_index"],
                                              [value.code_index for value in direct["raw_values"]])
                np.testing.assert_allclose(saved["observations"]["value_volts"],
                                           [value.value_volts for value in direct["observations"]], rtol=0, atol=0)
                self.assertEqual(len(saved["aperture_support"]), 2)

    def test_late_invalid_plan_preserves_initial_state_as_failed_run_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.h5"
            _, kwargs, _, _, _ = cases(self.world, path)
            original = kwargs["readout_window_factory"]
            def invalid():
                for window in original():
                    yield replace(window, end=.021) if window.index == 1 else window
            kwargs["readout_window_factory"] = invalid
            result = run_streaming_development_simulation(**kwargs)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["electrical_state_advanced"])
            self.assertEqual(result["failure"]["phase"], "whole-plan preflight")
            callback_path = Path(directory) / "callback_stopped.h5"
            _, callback_kwargs, _, _, _ = cases(self.world, callback_path)
            def stop_outer_budget():
                raise RuntimeError("outer budget fixture exhausted")
            stopped = run_streaming_development_simulation(**callback_kwargs,
                execution_check=stop_outer_budget)
            self.assertEqual(stopped["failure"]["phase"], "surface and hold preparation")
            self.assertEqual(stopped["failure"]["message"], "outer budget fixture exhausted")
            self.assertFalse(stopped["electrical_state_advanced"])
            self.assertFalse(callback_path.exists())
            with h5py.File(path, "r") as saved:
                self.assertTrue(saved.attrs["record_complete"])
                self.assertEqual(saved.attrs["simulation_status"], "failed")
                self.assertEqual(len(saved["raw_values"]), 0)
                np.testing.assert_array_equal(saved["initial_state_volts"][...], saved["final_state_volts"][...])
            interrupted_path = Path(directory) / "interrupted.h5"
            _, interrupted_kwargs, _, _, _ = cases(self.world, interrupted_path)
            def interrupt_after_first_window(runner):
                runner.advance_window()
                raise KeyboardInterrupt("fixture interruption after an actual advance")
            with patch.object(StreamingArrayAcquisition, "run", interrupt_after_first_window):
                with self.assertRaises(KeyboardInterrupt):
                    run_streaming_development_simulation(**interrupted_kwargs)
            # The interrupt still reaches the caller, after failed-run state
            # is saved and the exclusive writer is closed for inspection.
            with h5py.File(interrupted_path, "r") as saved:
                self.assertTrue(saved.attrs["record_complete"])
                self.assertEqual(saved.attrs["simulation_status"], "failed")
                self.assertGreater(len(saved["interval_evidence"]), 0)
                self.assertFalse(np.array_equal(saved["initial_state_volts"][...],
                                                saved["final_state_volts"][...]))

    def test_frozen_physical_noise_reaches_full_streamed_output_with_same_replayed_path(self):
        def noise_factory(*, array, check):
            layout = array.make_noise_layout(nodal_current_ports=(
                NodalCurrentPort("fixture_input_current", "source_00", None,
                                 "explicit_fixture_current", "fixture_source_00_input"),),
                contact_noise_role="fixture_other_ports_explicitly_zero",
                amplifier_input_noise_role=None, excluded_retained_contact_components=(),
                limits=NoiseLimits(10000, 4 * 1024 * 1024))
            spectrum = SpectralNoise(frequencies_hz=(1.,), quadrature_widths_hz=(1.,),
                port_ids=layout.port_ids, port_units=layout.port_units, latent_ids=("common",),
                sparse_terms=((0, 0, 0, .002),),
                quadratures=(LatentQuadrature(0, "common", 1., 0.),),
                time_window_seconds=(0., .02),
                limits=SpectralNoiseLimits(1, 1, 1, 10000, 1024 * 1024, 1000, 2000), check=check)
            return SpectralCircuitNoise(layout=layout, spectrum=spectrum,
                port_absolute_tolerances=(1e-4,) * layout.numeric_value_count, check=check)
        with tempfile.TemporaryDirectory() as directory:
            first_path, replay_path = Path(directory) / "noise.h5", Path(directory) / "noise_replayed.h5"
            eager, kwargs, _, _, _ = cases(self.world, first_path)
            first = run_streaming_development_simulation(**kwargs, noise_factory=noise_factory)
            repeated = run_streaming_development_simulation(**dict(kwargs,
                physical_actions=lambda: iter(eager["physical_actions"]),
                result_path=replay_path, noise_factory=noise_factory))
            for result in (first, repeated):
                self.assertEqual(result["status"], "complete", result)
                self.assertTrue(result["hold"]["physical_noise_hold_bound_applied"])
                self.assertTrue(result["readout"]["physical_noise_applied"])
                self.assertEqual(result["readout"]["actual_internal_conversions"], 2)
                self.assertNotEqual(result["accounting"]["physical_work"]["noise_current_source_interval_work_joule"], 0.)
                self.assertFalse(result["physical_noise"]["complete_device_noise_inventory_established"])
            self.assertEqual(first["accounting"]["physical_work"], repeated["accounting"]["physical_work"])
            with h5py.File(first_path, "r") as retained, h5py.File(replay_path, "r") as replay:
                for name in ("final_state_volts", "final_field_volts", "raw_values",
                             "observations", "interval_evidence", "aperture_support"):
                    np.testing.assert_array_equal(retained[name][...], replay[name][...])

    def test_saved_N4_requires_bindings_and_changes_only_decoded_observation_availability(self):
        # The contact bridge deliberately admits only the selected full grid.
        # These private fixtures are built only when this compute test runs.
        world = _build("N4", 2001, _Grid(), check=lambda: None,
                       max_elements=LIMIT, include_scales=False)
        paired = _build("B2", 2001, _Grid(), check=lambda: None,
                        max_elements=LIMIT, include_scales=False)
        dead_site = world.metadata["details"]["nuisance"]["dead_physical_row_zero_based"] * 64
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing_bindings.h5"
            _, incomplete, _, _, _ = cases(world, missing_path)
            refused = run_streaming_development_simulation(**incomplete)
            self.assertEqual(refused["status"], "failed")
            self.assertEqual(refused["failure"]["phase"], "whole-plan preflight")
            self.assertFalse(refused["electrical_state_advanced"])
            self.assertFalse(refused["nuisance_applied"])
            baseline_path, nuisance_path = Path(directory) / "paired_B2.h5", Path(directory) / "N4.h5"
            _, baseline, _, _, _ = cases(paired, baseline_path)
            _, selected, _, _, _ = cases(world, nuisance_path)
            original = selected["readout_window_factory"]
            def bound():
                for window in original():
                    index = window.index
                    yield replace(window, nuisance_bindings=(ObservationNuisanceBinding(
                        f"o:{index}", dead_site if index == 0 else (dead_site + 64) % 4096,
                        .004 if index == 0 else .015),))
            selected["readout_window_factory"] = bound
            expected = run_streaming_development_simulation(**baseline)
            actual = run_streaming_development_simulation(**selected)
            self.assertEqual((expected["status"], actual["status"]), ("complete", "complete"))
            self.assertTrue(actual["nuisance_applied"])
            self.assertFalse(actual["candidate_scored"])
            self.assertEqual(actual["readout"]["valid_retained_pixel_observations"], 1)
            self.assertEqual(actual["readout"]["nuisance_observations_processed"], 2)
            for name in ("actual_internal_conversions", "hardware_returned_samples", "output_bits"):
                self.assertEqual(actual["readout"][name], expected["readout"][name])
            with h5py.File(baseline_path, "r") as left, h5py.File(nuisance_path, "r") as right:
                for name in ("final_state_volts", "final_field_volts", "raw_values", "interval_evidence",
                             "aperture_support"):
                    np.testing.assert_array_equal(left[name][...], right[name][...])
                for name in ("observation_id", "site_id", "role", "clipped", "assigned_timestamp", "available_at"):
                    np.testing.assert_array_equal(left["observations"][name], right["observations"][name])
                self.assertEqual(right["observations"]["has_value"].tolist(), [False, True])
                self.assertEqual(right["observations"]["valid"].tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
