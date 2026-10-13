"""Focused checks of the parameterized return predicate, not real-data evidence.

Run only inside a coordinated Slurm allocation. No source recording, fitted
calibration, onset detector or forecasting model is used by these checks.
"""

from dataclasses import replace
import unittest

import numpy as np

from way_completed_return import (
    FrozenReturnReferences,
    ReturnAdmissions,
    StartContext,
    evaluate_completed_return,
)


ARRAY_INPUTS = (
    "positions_m",
    "object_motion",
    "plate_forces_n",
    "primary_summaries",
    "primary_history_valid",
    "object_motion_history_valid",
    "sample_valid",
)
NUMERIC_INPUTS = ARRAY_INPUTS[:4]
HISTORY_INPUTS = ARRAY_INPUTS[4:]


def supplied_window():
    """400 small factual rows; only ticks 50 through 299 support the cue."""
    count = 400
    positions = np.zeros((count, 4, 3))
    positions[:, 1, 0] = 0.25
    positions[:, 2, 0] = -0.25
    # Deliberately name a supplied feature unit: the predicate must neither
    # construct an object derivative nor assume the feature's physical units.
    motion_unit = "supplied_backward_feature_unit"
    released_force = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    return dict(
        evaluation_tick=300,
        first_tick=0,
        positions_m=positions,
        object_motion=np.zeros((count, 3)),
        plate_forces_n=np.broadcast_to(released_force, (count, 2, 3)).copy(),
        primary_summaries=np.full((count, 8), 0.5),
        primary_history_valid=np.ones((count, 8), dtype=bool),
        object_motion_history_valid=np.ones(count, dtype=bool),
        sample_valid=np.ones(count, dtype=bool),
        admissions=ReturnAdmissions(
            position_unit="m",
            force_unit="N",
            object_motion_unit=motion_unit,
            physical_units_admitted=True,
            backward_motion_admitted=True,
            primary_summaries_causal=True,
            calibration_admitted=True,
        ),
        references=FrozenReturnReferences(
            object_position_m=np.zeros(3),
            object_position_tolerance_m=np.full(3, 0.125),
            object_motion=np.zeros(3),
            object_motion_tolerance=np.full(3, 0.25),
            object_motion_unit=motion_unit,
            plate_force_n=released_force,
            plate_force_tolerance_n=np.full((2, 3), 0.125),
            primary_low=np.ones(8),
        ),
        context=StartContext(
            start_tick=300,
            start_authenticated=True,
            phase="initial_series",
            phase_history_known=True,
            series_start_tick=0,
            continuous_history_start_tick=0,
        ),
    )


class WayCompletedReturnTests(unittest.TestCase):
    def status(self, inputs):
        return evaluate_completed_return(**inputs).status

    def test_exact_250_prestart_ticks_ignore_prefix_current_and_future(self):
        inputs = supplied_window()
        result = evaluate_completed_return(**inputs)
        self.assertEqual(result.status, "supported")
        self.assertEqual((result.interval_start_tick, result.interval_stop_tick),
                         (50, 300))
        # Missing samples outside the exact support interval cannot affect it.
        for name in NUMERIC_INPUTS:
            inputs[name][:50] = np.nan
            inputs[name][300:] = np.inf
        for name in HISTORY_INPUTS:
            inputs[name][:50] = False
            inputs[name][300:] = False
        self.assertEqual(self.status(inputs), "supported")
        inputs["positions_m"][300:] = -1e100
        inputs["primary_summaries"][300:] = np.nan
        self.assertEqual(self.status(inputs), "supported")

        # No prefix or cue-row padding is needed if exactly 250 rows exist.
        exact = dict(inputs, first_tick=50)
        for name in ARRAY_INPUTS:
            exact[name] = inputs[name][50:300].copy()
        self.assertEqual(self.status(exact), "supported")
        short = dict(exact, first_tick=51)
        for name in ARRAY_INPUTS:
            short[name] = exact[name][1:]
        self.assertEqual(self.status(short), "unknown")

        # Both ends are required, even if the other 249 samples are quiet.
        for tick in (50, 299):
            with self.subTest(required_tick=tick):
                broken = supplied_window()
                broken["primary_summaries"][tick, 0] = 2.0
                self.assertEqual(self.status(broken), "not_supported")

    def test_both_digit_clearances_and_every_primary_path_are_required(self):
        inputs = supplied_window()
        inputs["positions_m"][:, 1, 0] = 0.15
        inputs["positions_m"][:, 2, 0] = -0.15
        inputs["primary_summaries"][:] = 1.0
        self.assertEqual(self.status(inputs), "supported")
        for digit in (1, 2):
            with self.subTest(digit=digit):
                broken = supplied_window()
                broken["positions_m"][177, digit] = [0.149, 0.0, 0.0]
                self.assertEqual(self.status(broken), "not_supported")
        for path in range(8):
            with self.subTest(primary_path=path):
                broken = supplied_window()
                broken["primary_summaries"][177, path] = 1.01
                self.assertEqual(self.status(broken), "not_supported")

    def test_object_and_individual_plate_coordinate_checks_do_not_cancel(self):
        inputs = supplied_window()
        # Coordinate-wise bounds permit the corner of the tolerance box.
        inputs["positions_m"][:] += 0.125
        inputs["object_motion"][:] = 0.25
        inputs["plate_forces_n"][:] += 0.125
        self.assertEqual(self.status(inputs), "supported")

        for coordinate in range(3):
            with self.subTest(object_position_coordinate=coordinate):
                broken = supplied_window()
                # Move all sensors together to preserve the digit distances.
                broken["positions_m"][177, :, coordinate] += 0.126
                self.assertEqual(self.status(broken), "not_supported")
            with self.subTest(object_motion_coordinate=coordinate):
                broken = supplied_window()
                broken["object_motion"][177, coordinate] = -0.251
                self.assertEqual(self.status(broken), "not_supported")
            for plate in range(2):
                with self.subTest(plate=plate, coordinate=coordinate):
                    broken = supplied_window()
                    broken["plate_forces_n"][177, plate, coordinate] -= 0.126
                    self.assertEqual(self.status(broken), "not_supported")

        opposite = supplied_window()
        opposite["plate_forces_n"][177, 0, 2] += 1.0
        opposite["plate_forces_n"][177, 1, 2] -= 1.0
        self.assertEqual(self.status(opposite), "not_supported")

    def test_initial_entry_and_strict_reach_end_return_start_order(self):
        initial = supplied_window()
        self.assertIsNone(initial["context"].prior_reach_tick)
        self.assertIsNone(initial["context"].end_tick)
        self.assertEqual(self.status(initial), "supported")
        inputs = supplied_window()
        inputs["context"] = replace(
            inputs["context"], phase="after_reach", prior_reach_tick=10,
            end_tick=20, end_authenticated=True,
        )
        self.assertEqual(self.status(inputs), "supported")
        for end in (9, 10, 50, 60, 300):
            with self.subTest(end_tick=end):
                broken = dict(inputs)
                broken["context"] = replace(inputs["context"], end_tick=end)
                self.assertEqual(self.status(broken), "not_supported")
        for missing in (
            {"prior_reach_tick": None},
            {"end_tick": None},
            {"end_authenticated": False},
        ):
            with self.subTest(missing=missing):
                broken = dict(inputs)
                broken["context"] = replace(inputs["context"], **missing)
                self.assertEqual(self.status(broken), "unknown")

    def test_late_evaluation_and_gap_or_series_history_cannot_restore_entry(self):
        inputs = supplied_window()
        # Quiet samples after the cue cannot create late first arming.
        inputs["evaluation_tick"] = 350
        self.assertEqual(self.status(inputs), "not_supported")
        # Off-start queries must not inspect an unavailable or future window.
        for evaluation_tick, expected in ((299, "unknown"), (350, "not_supported")):
            with self.subTest(off_start=evaluation_tick):
                outside_domain = supplied_window()
                outside_domain["evaluation_tick"] = evaluation_tick
                for name in ARRAY_INPUTS:
                    outside_domain[name] = object()
                self.assertEqual(self.status(outside_domain), expected)
        for context_change in (
            {"continuous_history_start_tick": 51},
            {"series_start_tick": 51, "continuous_history_start_tick": 51},
            {"continuous_history_start_tick": None},
            {"series_start_tick": None},
            {"phase": None},
            {"phase_history_known": False},
        ):
            with self.subTest(context_change=context_change):
                broken = supplied_window()
                broken["context"] = replace(broken["context"], **context_change)
                self.assertEqual(self.status(broken), "unknown")
        # A complete numeric window does not preserve pre-gap reach/end state.
        after_gap = supplied_window()
        after_gap["context"] = replace(
            after_gap["context"], phase="after_reach", prior_reach_tick=10,
            end_tick=20, end_authenticated=True, continuous_history_start_tick=30,
        )
        self.assertEqual(self.status(after_gap), "unknown")
        later = supplied_window()
        later["context"] = replace(later["context"], phase=None,
                                   continuous_history_start_tick=50)
        self.assertEqual(self.status(later), "unknown")

    def test_missing_prerequisites_and_nonfinite_support_take_unknown_priority(self):
        def with_veto():
            inputs = supplied_window()
            inputs["primary_summaries"][100, 0] = 2.0
            return inputs

        for missing in ("admissions", "references", "context"):
            with self.subTest(missing=missing):
                inputs = with_veto()
                inputs[missing] = None
                self.assertEqual(self.status(inputs), "unknown")
        wrong_grid = with_veto()
        wrong_grid["sample_rate_hz"] = 1000
        self.assertEqual(self.status(wrong_grid), "unknown")
        for admission_change in (
            {"position_unit": None}, {"position_unit": "mm"},
            {"force_unit": None}, {"force_unit": "source_counts"},
            {"object_motion_unit": None}, {"object_motion_unit": "other_unit"},
            {"physical_units_admitted": False},
            {"backward_motion_admitted": False},
            {"primary_summaries_causal": False},
            {"calibration_admitted": False},
        ):
            with self.subTest(admission_change=admission_change):
                inputs = with_veto()
                inputs["admissions"] = replace(inputs["admissions"],
                                                **admission_change)
                self.assertEqual(self.status(inputs), "unknown")
        for field in (
            "object_position_m", "object_position_tolerance_m", "object_motion",
            "object_motion_tolerance", "object_motion_unit", "plate_force_n",
            "plate_force_tolerance_n", "primary_low",
        ):
            with self.subTest(missing_reference=field):
                inputs = with_veto()
                inputs["references"] = replace(inputs["references"], **{field: None})
                self.assertEqual(self.status(inputs), "unknown")
        for context_change in ({"start_tick": None}, {"start_authenticated": False}):
            with self.subTest(context_change=context_change):
                inputs = with_veto()
                inputs["context"] = replace(inputs["context"], **context_change)
                self.assertEqual(self.status(inputs), "unknown")
        for name in NUMERIC_INPUTS:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(nonfinite_input=name, value=value):
                    inputs = with_veto()
                    inputs[name][177] = value
                    self.assertEqual(self.status(inputs), "unknown")
        # Finite inputs can still produce an unrepresentable distance. That
        # missing numerical result has priority over the observed low veto.
        overflow = with_veto()
        largest = np.finfo(np.float64).max
        overflow["positions_m"][177, 0, 0] = largest
        overflow["positions_m"][177, 1, 0] = -largest
        self.assertEqual(self.status(overflow), "unknown")
        for path in range(8):
            with self.subTest(missing_primary_history=path):
                inputs = with_veto()
                inputs["primary_history_valid"][177, path] = False
                self.assertEqual(self.status(inputs), "unknown")
        for name in ("object_motion_history_valid", "sample_valid"):
            with self.subTest(missing_history=name):
                inputs = with_veto()
                inputs[name][177] = False
                self.assertEqual(self.status(inputs), "unknown")


if __name__ == "__main__":
    unittest.main()
