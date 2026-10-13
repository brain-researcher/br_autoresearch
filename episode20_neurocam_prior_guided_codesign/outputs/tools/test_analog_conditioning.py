"""Focused conditional-macromodel algebra; no selected hardware constants.

Run only in a coordinator-owned compute allocation. Joint state propagation
and retained aperture history are covered by the solver/assembly integration
fixtures, not approximated by a disconnected filter in this module.
"""

from dataclasses import FrozenInstanceError, replace
import unittest

import numpy as np

from analog_conditioning import AnalogConditioning, SourceConditioning


def channel(**changes):
    values = dict(gain=2., offset_volts=.25, drive_lower_volts=-1., drive_upper_volts=1.5,
                  input_capacitance_farad=.2, input_conductance_siemens=.3,
                  buffer_capacitance_farad=2., buffer_conductance_siemens=3.,
                  output_conductance_siemens=4., filter_capacitance_farad=5.,
                  filter_leak_conductance_siemens=.1,
                  initial_buffer_volts=.4, initial_filter_volts=-.2)
    values.update(changes)
    return SourceConditioning(**values)


def conditioning(channels=None):
    return AnalogConditioning([channel()] * 64 if channels is None else channels,
        "additional_fixed_shunt_excludes_conditioning",
        "additional_fixed_capacitance_excludes_conditioning", "common")


def stamp(model):
    return model.stamp(state_count=4312, source_node_offset=4096,
                       first_conditioning_state=4160, voltage_reference_id="common")


class AnalogConditioningTests(unittest.TestCase):
    def test_all_source_stamp_owns_loads_and_preserves_mux_axis(self):
        channels = [channel(gain=source + 1.) for source in range(64)]
        model = conditioning(channels)
        channels.clear()
        result = stamp(model)
        self.assertEqual(model.state_count, 128)
        self.assertEqual(result.controlled_drives.source_nodes, tuple(range(4096, 4160)))
        self.assertEqual(result.controlled_drives.buffer_nodes, tuple(range(4160, 4224)))
        self.assertEqual(result.controlled_drives.buffer_conductance_siemens, (3.,) * 64)
        self.assertEqual(result.output_nodes, tuple(range(4224, 4288)))
        self.assertEqual(len(set(result.state_ids)), 128)
        self.assertEqual(result.state_ids[63:65], ("conditioning_buffer_63", "conditioning_filter_00"))
        self.assertEqual(result.source_capacitance_farad, (.2,) * 64)
        self.assertEqual(result.source_conductance_siemens, (.3,) * 64)
        self.assertEqual(result.added_capacitance_farad, (2.,) * 64 + (5.,) * 64)
        self.assertEqual(result.added_shunts_siemens, (3.,) * 64 + (.1,) * 64)
        self.assertEqual(result.initial_volts, (.4,) * 64 + (-.2,) * 64)
        self.assertEqual(result.resistor_edges,
                         tuple((4160 + source, 4224 + source, 4.) for source in range(64)))
        # Twenty-four downstream hold states are untouched by this stamp.
        self.assertEqual(result.controlled_drives.state_count - max(result.output_nodes) - 1, 24)
        self.assertEqual(result.controlled_drives.numeric_storage_bytes, 7 * 64 * 8)
        with self.assertRaises(FrozenInstanceError):
            model.channels = ()
        with self.assertRaises(FrozenInstanceError):
            result.controlled_drives._state_count = 1
        for kwargs in ({"source_load_role": "included"}, {"source_capacitance_role": "included"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(model, **kwargs)

    def test_saturated_target_signed_terminal_work_and_sparse_jacobian(self):
        result = stamp(conditioning())
        drive = result.controlled_drives
        volts = np.zeros(drive.state_count)
        # Lower saturation, interior, upper saturation and exact clip kinks.
        values = (-2., 0., 2., -.625, .625)
        source_values = np.array([values[source % len(values)] for source in range(64)])
        buffer_values = np.array([2. if source % 2 == 0 else -.5 for source in range(64)])
        volts[list(drive.source_nodes)] = source_values
        volts[list(drive.buffer_nodes)] = buffer_values
        before = volts.copy()
        evaluation = drive.evaluate(volts)
        targets = np.array([min(1.5, max(-1., 2. * value + .25)) for value in source_values])
        np.testing.assert_array_equal(evaluation.current_amp, 3. * targets)
        self.assertEqual(evaluation.node_indices, drive.buffer_nodes)
        expected_source = sum(3. * target * (target - buffer)
                              for target, buffer in zip(targets, buffer_values))
        expected_heat = sum(3. * (target - buffer) ** 2
                            for target, buffer in zip(targets, buffer_values))
        self.assertAlmostEqual(evaluation.controlled_voltage_source_power_watt, expected_source)
        self.assertAlmostEqual(evaluation.controlled_branch_dissipation_watt, expected_heat)
        self.assertAlmostEqual(evaluation.controlled_voltage_source_power_watt
                               - evaluation.controlled_branch_dissipation_watt,
                               evaluation.controlled_nodal_current_power_watt
                               - evaluation.buffer_shunt_power_watt)
        np.testing.assert_array_equal(volts, before)
        # Output is not reset/clamped. With f=b, the persistent buffer's local
        # derivative is gb*(target-b)/Cb, including recovery from above a rail.
        recovery_rate = (evaluation.current_amp[2] - 3. * buffer_values[2]) / 2.
        self.assertEqual(buffer_values[2], 2.)
        self.assertEqual(targets[2], 1.5)
        self.assertEqual(recovery_rate, -.75)
        # Backdriving an ideal controlled port may return signed terminal work.
        backdriven = volts.copy()
        backdriven[list(drive.source_nodes)] = 2.
        backdriven[list(drive.buffer_nodes)] = 2.
        self.assertLess(drive.evaluate(backdriven).controlled_voltage_source_power_watt, 0.)

        direction = np.zeros(drive.state_count)
        direction[list(drive.source_nodes)] = 2.
        direction[list(drive.buffer_nodes)] = 100.  # Damping is in passive G, not this J.
        expected_j = np.array([12. if -1. < 2. * value + .25 < 1.5 else 0.
                               for value in source_values])
        np.testing.assert_array_equal(drive.jacobian_product(volts, direction), expected_j)
        # A returned current vector is not an alias of retained configuration.
        evaluation.current_amp[:] = 999.
        np.testing.assert_array_equal(drive.evaluate(volts).current_amp, 3. * targets)

    def test_invalid_physics_reference_axes_and_nonfinite_drive_refuse(self):
        for changes in ({"buffer_capacitance_farad": 0.}, {"output_conductance_siemens": 0.},
                        {"filter_leak_conductance_siemens": -1.},
                        {"drive_lower_volts": 1.5}, {"gain": float("nan")}, {"gain": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                channel(**changes)
        for channels in ([channel()] * 63, [channel()] * 65):
            with self.assertRaises(ValueError):
                conditioning(channels)
        model = conditioning()
        for changes in ({"voltage_reference_id": "other"}, {"first_conditioning_state": 4159},
                        {"state_count": 4287}, {"source_node_offset": True}):
            arguments = dict(state_count=4312, source_node_offset=4096,
                             first_conditioning_state=4160, voltage_reference_id="common")
            arguments.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                model.stamp(**arguments)
        drive = stamp(model).controlled_drives
        for values in (np.zeros(64), np.full(drive.state_count, float("nan")),
                       np.ones(drive.state_count, dtype=complex)):
            with self.assertRaises(ValueError):
                drive.evaluate(values)
        overflow = np.zeros(drive.state_count)
        overflow[drive.source_nodes[0]] = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, "representable"):
            drive.evaluate(overflow)
        with self.assertRaises(ValueError):
            drive.jacobian_product(np.zeros(drive.state_count), np.zeros(64))


if __name__ == "__main__":
    unittest.main()
