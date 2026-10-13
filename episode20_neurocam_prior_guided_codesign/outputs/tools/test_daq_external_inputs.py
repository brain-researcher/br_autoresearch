"""Frontend-only fixtures; run in coordinated compute, never on login.

The small circuit numbers below are artificial algebra fixtures, not physical
parameter choices or a proposed NeuroCam acquisition configuration.
"""
from dataclasses import FrozenInstanceError
import unittest

from daq_mux_hold import (DAQGroupDrive, ExternalVoltageInput,
                          PassiveDAQFrontend, PassiveMuxHoldGroup)


class ExternalDAQInputTests(unittest.TestCase):
    def frontend(self, *groups):
        return PassiveDAQFrontend(groups, "additional_fixed_shunt_excludes_frontend")

    def mixed(self, adc_id="adc0"):
        return PassiveMuxHoldGroup(adc_id, (1, 3), 2., 0., (2., 4.), (.2, .4), .1,
            (ExternalVoltageInput("marker", 3., .3), ExternalVoltageInput("reference", 5., 0.)))

    def test_legacy_edges_and_external_only_groups_keep_neural_axis_and_path_counts(self):
        legacy = PassiveMuxHoldGroup("legacy", (0, 63), 2., 0., (2., 4.), (.2, .4), .1)
        frontend = self.frontend(legacy)
        frontend.validate_source_count(64)
        self.assertEqual(frontend.resistor_edges(selected_sources={"legacy": 63},
            source_node_offset=4096, first_hold_state=4160),
            ((4096, 4160, .2), (4159, 4160, 4.)))
        self.assertEqual(frontend.external_drives(selected_sources={"legacy": None},
            first_hold_state=4160, voltage_reference_id="common"), ())
        self.assertEqual((frontend.hold_state_count, frontend.switch_branch_count), (1, 2))
        bad = PassiveMuxHoldGroup("bad", (64,), 2., 0., (2.,), (.2,), .1)
        with self.assertRaisesRegex(ValueError, "exceeds the assembled array"):
            self.frontend(bad).validate_source_count(64)

        external = PassiveMuxHoldGroup("marker_adc", (), 2., 0., (), (), .1,
                                      (ExternalVoltageInput("marker", 3., 0.),))
        frontend = self.frontend(legacy, external)
        selection = {"legacy": 0, "marker_adc": DAQGroupDrive("marker", (("marker", 7.),), "common")}
        frontend.validate_source_count(64)
        self.assertEqual((frontend.hold_state_count, frontend.switch_branch_count), (2, 3))
        self.assertEqual(frontend.external_drives(selected_sources=selection,
            first_hold_state=4160, voltage_reference_id="common"), ((4161, 3., 7.),))
        self.assertEqual(len(frontend.resistor_edges(selected_sources=selection,
            source_node_offset=4096, first_hold_state=4160)), 2)
        with self.assertRaises(ValueError):
            PassiveMuxHoldGroup("empty", (), 2., 0., (), (), .1)

    def test_selected_and_off_external_branches_are_separate_from_neural_edges(self):
        frontend = self.frontend(self.mixed())
        for selected, expected_edges, expected_drives in (
            (1, ((101, 200, 2.), (103, 200, .4)), ((200, .3, 7.),)),
            ("marker", ((101, 200, .2), (103, 200, .4)), ((200, 3., 7.),)),
            ("reference", ((101, 200, .2), (103, 200, .4)),
             ((200, .3, 7.), (200, 5., -2.))),
            (None, ((101, 200, .2), (103, 200, .4)), ((200, .3, 7.),)),
        ):
            selection = {"adc0": DAQGroupDrive(selected, (("marker", 7.), ("reference", -2.)), "common")}
            with self.subTest(selected=selected):
                self.assertEqual(frontend.resistor_edges(selected_sources=selection,
                    source_node_offset=100, first_hold_state=200), expected_edges)
                self.assertEqual(frontend.external_drives(selected_sources=selection,
                    first_hold_state=200, voltage_reference_id="common"), expected_drives)
        self.assertEqual(frontend.switch_branch_count, 4)  # Includes zero-off reference path.

    def test_owned_drive_and_declared_ports_reject_incomplete_or_invalid_bindings(self):
        values = [["marker", 7], ["reference", -2]]
        drive = DAQGroupDrive("marker", values, "common")
        values[0][1] = 999
        values.append(["unrelated", 1])
        self.assertEqual(drive.external_volts, (("marker", 7.), ("reference", -2.)))
        with self.assertRaises(FrozenInstanceError):
            drive.selected_input = None
        declarations = [ExternalVoltageInput("marker", 3., .3)]
        group = PassiveMuxHoldGroup("adc0", (), 2., 0., (), (), .1, declarations)
        declarations.clear()
        self.assertEqual(len(group.external_inputs), 1)

        frontend = self.frontend(self.mixed())
        invalid = [None, 1,
            DAQGroupDrive(None, (("marker", 7.),), "common"),
            DAQGroupDrive(None, (("marker", 7.), ("reference", -2.), ("extra", 0.)), "common"),
            DAQGroupDrive("not_declared", drive.external_volts, "common"),
            DAQGroupDrive(2, drive.external_volts, "common")]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                frontend.resistor_edges(selected_sources={"adc0": value}, source_node_offset=100, first_hold_state=200)
        for bad_values in ((("marker", float("nan")),), (("marker", float("inf")),),
                           (("marker", True),), (("marker", 1.), ("marker", 2.))):
            with self.subTest(values=bad_values), self.assertRaises(ValueError):
                DAQGroupDrive(None, bad_values, "common")
        for selected in (True, -1, 1.5, ""):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                DAQGroupDrive(selected, (("marker", 1.),), "common")
        for track, off in ((0., 0.), (1., -1.), (float("inf"), 0.)):
            with self.subTest(track=track, off=off), self.assertRaises(ValueError):
                ExternalVoltageInput("marker", track, off)
        with self.assertRaises(ValueError):
            PassiveMuxHoldGroup("duplicate", (), 2., 0., (), (), .1,
                (ExternalVoltageInput("marker", 3., 0.), ExternalVoltageInput("marker", 4., 0.)))

    def test_common_reference_and_global_port_identity_checked_even_for_zero_off_paths(self):
        first = PassiveMuxHoldGroup("a", (), 2., 0., (), (), .1,
                                   (ExternalVoltageInput("marker", 3., .3),))
        second = PassiveMuxHoldGroup("b", (), 2., 0., (), (), .1,
                                    (ExternalVoltageInput("marker", 5., 0.),))
        frontend = self.frontend(first, second)
        selection = {"a": DAQGroupDrive("marker", (("marker", 7.),), "common"),
                     "b": DAQGroupDrive(None, (("marker", 7.),), "common")}
        self.assertEqual(frontend.external_drives(selected_sources=selection,
            first_hold_state=10, voltage_reference_id="common"), ((10, 3., 7.),))
        selection["b"] = DAQGroupDrive(None, (("marker", 8.),), "common")
        with self.assertRaisesRegex(ValueError, "same voltage"):
            frontend.external_drives(selected_sources=selection, first_hold_state=10,
                                     voltage_reference_id="common")
        selection["b"] = DAQGroupDrive(None, (("marker", 7.),), "other_reference")
        with self.assertRaisesRegex(ValueError, "voltage reference differs"):
            frontend.external_drives(selected_sources=selection, first_hold_state=10,
                                     voltage_reference_id="common")
        selection.pop("b")
        with self.assertRaisesRegex(ValueError, "Every physical DAQ group"):
            frontend.external_drives(selected_sources=selection, first_hold_state=10,
                                     voltage_reference_id="common")


if __name__ == "__main__":
    unittest.main()
