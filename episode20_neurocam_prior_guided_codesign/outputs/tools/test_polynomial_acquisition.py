"""Full-array cubic provider/acquisition seams, not a hardware/noise recipe."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
from array_acquisition import (AcquisitionExecutionError, AcquisitionLimits, ArrayAcquisition,
                               ArrayInputBinding, ProviderInterval)
from daq_mux_hold import external_daq_branch_id
from physical_noise_ports import ExternalDAQCurrentPort, NoiseLimits, NodalCurrentPort
from polynomial_noise import NoisePolynomial
from streaming_acquisition_readout import ReadoutWindow, StreamingReadoutLimits
from streaming_array_acquisition import (ArrayWindow, StreamPhysicalInterval,
                                         StreamingArrayAcquisition, StreamingArrayLimits)
from test_conditioned_cpe_array import ADC, REF, fixture as conditioned_fixture, selections, solver_controls


KEYS = (1.004, 1.010)
EAGER_LIMITS = AcquisitionLimits(4, 16, 2, 2, 2, 1024**2)
STREAM_LIMITS = StreamingArrayLimits(4, 1, 4, 8, 4, 16, 8, 1024**2, 128, 16_000_000)
READOUT_LIMITS = StreamingReadoutLimits(2, 2, 2, 16, 2, 2, 2, 2, 1024**2, 2)
WORK_FIELDS = ("field_source_interval_work_joule", "noise_current_source_interval_work_joule",
               "noise_access_voltage_source_interval_work_joule", "daq_external_source_interval_work_joule",
               "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule")


def fixture():
    array = conditioned_fixture(initial_source=0., external=True)
    layout = array.make_noise_layout(nodal_current_ports=(
        NodalCurrentPort("source_i", "source_00", None, "conditional_current", "fixture_current_i"),
        NodalCurrentPort("source_j", "source_00", None, "conditional_current", "fixture_current_j")),
        contact_noise_role="conditional_finite_band", amplifier_input_noise_role="conditional_input_voltage",
        external_daq_current_ports=(ExternalDAQCurrentPort("external_i", external_daq_branch_id(ADC, "G1"),
            ADC, "G1", "daq_hold_" + ADC, "conditional_switch_current"),),
        excluded_retained_contact_components=(), limits=NoiseLimits(100000, 8 * 1024**2))
    first = ProviderInterval(1., 1.005, 0, KEYS[0], REF, selections("G1", .4), ((4096, .03),))
    continued = replace(first, start=1.005, end=1.008, injected_charge_coulomb=())
    following = ProviderInterval(1.008, 1.012, 1, KEYS[1], REF, selections(0, -.2), ())
    transfer = QuantizerTransfer(-1., 1., (-.5, 0., .5), (-.75, -.25, .25, .75))
    raw = RawConversion("r:0", ADC, "hold", "signal", 1.002, 1.010, 1.010, 1.002, transfer, True)
    word = ReturnedSample("s:0", ("r:0",), (1.,), 1., 1.002, 1.012, None)
    observation = Observation("o:0", "fixture_pixel", "pixel", ("s:0",), (1.,), 1., 1.002, 1.015, 16, 0)
    readouts = (ReadoutWindow(0, 1., 1.005, (raw,), (word,), (observation,), (("r:0", 1),), (("s:0", 1),)),
                ReadoutWindow(1, 1.005, 1.012, (), (), (), (), ()))
    windows = (ArrayWindow(0, 1., 1.005, (StreamPhysicalInterval(0, first),), readouts[0]),
               ArrayWindow(1, 1.005, 1.012, (StreamPhysicalInterval(0, continued),
                                             StreamPhysicalInterval(1, following)), readouts[1]))
    return array, layout, windows


def polynomial(layout, key, *, drift=False, bad_support=False):
    rows = np.zeros((4, layout.numeric_value_count))
    rows[:, layout.port_ids.index("source_i")] = (.0625, 1., -2., 16.)
    rows[:, layout.port_ids.index("amplifier_input_voltage:00")] = (.03, .8, -2., 10.)
    rows[:, layout.port_ids.index("external_i")] = (.02, -.7, 5., -20.)
    if drift:
        # Equal-and-opposite ORIGINAL current coefficients cancel on the node,
        # including at every time; they still change the owned physical ports.
        rows[3, layout.port_ids.index("source_i")] += 32.
        rows[3, layout.port_ids.index("source_j")] -= 32.
    start, end = (1., 1.008) if key == KEYS[0] else (1.008, 1.012)
    if bad_support and key == KEYS[1]:
        end = 1.011
    return NoisePolynomial(layout, rows, key, start, end)


def provider(layout, *, calls=None, changed=None, bad_support=False):
    def supply(key):
        if calls is not None:
            calls.append(key)
        return polynomial(layout, key, drift=bool(changed and changed[0] and key == KEYS[1]),
                          bad_support=bad_support)
    supply.sample_numeric_bytes = 32 * layout.numeric_value_count + 24
    return supply


def arguments(array, windows, supplied):
    return dict(array=array, field_provider=lambda key: np.zeros((64, 64)), noise_provider=supplied,
        input_bindings=(ArrayInputBinding("hold", ADC, "hold", None, ADC),),
        acquisition_window=(1., 1.012), final_publication_horizon=1.015,
        controls=solver_controls(relative_tolerance=2e-6, voltage_atol=1e-9,
            charge_atol_coulomb=1e-10, energy_atol_joule=1e-9,
            max_operator_products=1_000_000, max_linear_iterations=100000), check=lambda: None)


def eager_arguments(array, windows, supplied):
    return dict(**arguments(array, windows, supplied),
        physical_intervals=tuple(fragment.interval for window in windows for fragment in window.physical_intervals),
        conversions=tuple(item for window in windows for item in window.readout_window.conversions),
        returned_samples=tuple(item for window in windows for item in window.readout_window.returned_samples),
        observations=tuple(item for window in windows for item in window.readout_window.observations), limits=EAGER_LIMITS)


def stream_arguments(array, windows, supplied, *, raw=None, observations=None):
    return dict(**arguments(array, windows, supplied), window_factory=lambda: iter(windows),
        limits=STREAM_LIMITS, readout_limits=READOUT_LIMITS,
        on_raw=(lambda _: None) if raw is None else raw.append,
        on_observation=(lambda _: None) if observations is None else observations.append,
        on_interval=lambda _: None)


class PolynomialAcquisitionTests(unittest.TestCase):
    def test_same_original_cubic_survives_storage_aperture_and_mux_seams(self):
        array, layout, windows = fixture()
        supplied = provider(layout)
        raw, observations, physical = [], [], []
        original = array.advance_row
        def capture(**kwargs):
            result = original(**kwargs)
            physical.append((kwargs["interval_start_seconds"], kwargs["noise_sample"], result))
            return result
        runner = StreamingArrayAcquisition(**stream_arguments(array, windows, supplied,
            raw=raw, observations=observations))
        with patch.object(array, "advance_row", side_effect=capture):
            runner.advance_window()
            self.assertFalse(raw)
            self.assertFalse(observations)
            report = runner.run()
        self.assertEqual(array.resources.total_states, 4312)
        self.assertEqual(array.resources.daq_hold_states, 24)
        self.assertEqual([start for start, _, _ in physical], [1., 1.002, 1.005, 1.008, 1.010])
        self.assertTrue(all((sample.origin_seconds, sample.support_start_seconds, sample.support_end_seconds)
            == (KEYS[0], 1., 1.008) for _, sample, _ in physical[:3]))
        for _, sample, _ in physical[:3]:
            np.testing.assert_array_equal(sample.numeric_array, physical[0][1].numeric_array)
        self.assertGreater(physical[0][2].supplied_charge_boundary_work_joule, 0.)
        self.assertTrue(all(result.supplied_charge_boundary_work_joule == 0. for _, _, result in physical[1:]))
        self.assertEqual((report["physical_input_events"], report["physical_fragments"]), (2, 3))
        self.assertEqual((report["actual_internal_conversions"], report["hardware_returned_samples"],
                          report["retained_pixel_observations"]), (1, 1, 1))
        self.assertEqual(report["live_physical_noise_numeric_bytes"], supplied.sample_numeric_bytes)
        self.assertNotEqual(report["physical_work"]["noise_current_source_interval_work_joule"], 0.)
        eager, eager_layout, eager_windows = fixture()
        eager_physical = []
        original_eager = eager.advance_row
        def capture_eager(**kwargs):
            result = original_eager(**kwargs)
            eager_physical.append(result)
            return result
        with patch.object(eager, "advance_row", side_effect=capture_eager):
            expected = ArrayAcquisition(**eager_arguments(eager, eager_windows, provider(eager_layout))).run()
        np.testing.assert_array_equal(array.volts, eager.volts)
        self.assertEqual(tuple(raw), expected["raw_values"])
        self.assertEqual(tuple(observations), expected["observations"])
        for name in WORK_FIELDS:
            self.assertAlmostEqual(report["physical_work"][name], sum(getattr(result, name) for result in eager_physical), places=14)
            self.assertAlmostEqual(report["physical_work"][name], sum(getattr(result, name) for _, _, result in physical), places=14)

    def test_late_support_and_canceling_original_coefficient_changes_refuse_before_advance(self):
        array, layout, windows = fixture()
        before = array.volts, array.previous_field_volts
        with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
            with self.assertRaisesRegex(ValueError, "support"):
                ArrayAcquisition(**eager_arguments(array, windows, provider(layout, bad_support=True)))
            with self.assertRaisesRegex(ValueError, "support"):
                StreamingArrayAcquisition(**stream_arguments(array, windows, provider(layout, bad_support=True)))
            advance.assert_not_called()
        self.assertEqual((array.volts, array.previous_field_volts), before)
        normal, changed = polynomial(layout, KEYS[1]), polynomial(layout, KEYS[1], drift=True)
        context = array.daq_frontend.external_branch_drives(selected_sources=selections(0, -.2),
            first_hold_state=len(array.state_ids) - 24, voltage_reference_id=REF)
        for left, right in zip(normal.numeric_array, changed.numeric_array):
            np.testing.assert_array_equal(layout.project_current(layout.sample_from_values(left), external_branch_drives=context),
                layout.project_current(layout.sample_from_values(right), external_branch_drives=context))
        drifting = [False]
        runner = StreamingArrayAcquisition(**stream_arguments(array, windows, provider(layout, changed=drifting)))
        runner.advance_window()
        drifting[0] = True
        before = array.volts, array.previous_field_volts
        with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
            with self.assertRaises(AcquisitionExecutionError):
                runner.advance_window()
            advance.assert_not_called()
        self.assertEqual((array.volts, array.previous_field_volts), before)
        self.assertEqual(runner.failure["phase"], "window replay verification")
        with self.assertRaises(AcquisitionExecutionError):
            runner.run()

    def test_original_four_rows_and_support_are_charged_to_live_numeric_caps(self):
        array, layout, windows = fixture()
        calls = []
        supplied = provider(layout, calls=calls)
        self.assertEqual(polynomial(layout, KEYS[0]).sample_numeric_bytes, supplied.sample_numeric_bytes)
        with patch.object(array, "advance_row", wraps=array.advance_row) as advance:
            with self.assertRaisesRegex(ValueError, "numeric-byte"):
                ArrayAcquisition(**{**eager_arguments(array, windows, supplied), "limits":
                    replace(EAGER_LIMITS, max_stored_plan_numeric_bytes=supplied.sample_numeric_bytes - 1)})
            with self.assertRaisesRegex(ValueError, "numeric-byte"):
                StreamingArrayAcquisition(**{**stream_arguments(array, windows, supplied), "limits":
                    replace(STREAM_LIMITS, max_stored_window_numeric_bytes=supplied.sample_numeric_bytes - 1)})
            self.assertFalse(calls)
            supplied.sample_numeric_bytes = layout.sample_numeric_bytes  # The old one-row declaration is insufficient.
            with self.assertRaisesRegex(ValueError, "declared numeric storage"):
                ArrayAcquisition(**eager_arguments(array, windows, supplied))
            with self.assertRaisesRegex(ValueError, "declared numeric storage"):
                StreamingArrayAcquisition(**stream_arguments(array, windows, supplied))
            advance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
