"""Five focused algebra regressions; coordinator-owned Slurm execution only."""
from dataclasses import replace
import unittest

import numpy as np

from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
from slot_capacity import (RIDGE_GRID, SlotCapacityCurve, SlotCapacityResult,
                           admissible_capacity_pairs, slot_capacity_curves)


def direct_traces(base, slot, weights, penalties, ridge):
    """Independent full augmented least-squares influence, including base refit."""
    zbase = np.sqrt(weights)[:, None] * base
    zslot = np.sqrt(weights)[:, None] * slot
    design = np.column_stack((zbase, zslot))
    full_penalty = np.concatenate((penalties, np.full(slot.shape[1], ridge)))
    augmented = np.vstack((design, np.diag(np.sqrt(full_penalty))))
    influence = np.linalg.pinv(augmented, rcond=1e-12)[:, :len(base)]
    total = float(np.trace(design @ influence))
    term = float(np.trace(zslot @ influence[base.shape[1]:]))
    base_augmented = np.vstack((zbase, np.diag(np.sqrt(penalties))))
    base_influence = np.linalg.pinv(base_augmented, rcond=1e-12)[:, :len(base)]
    base_df = float(np.trace(zbase @ base_influence))
    return term, total - base_df, total


def basis(lags):
    return np.column_stack((np.ones(len(lags)), lags))


class SlotCapacityTests(unittest.TestCase):
    def test_penalized_base_term_edf_differs_from_incremental_model_edf(self):
        result = slot_capacity_curves(base_design=[[1.]], slot_designs={"added": [[1.]]},
                                      sample_weights=[1.], base_penalties=[1.],
                                      base_rank_relative_tolerance=1e-12)
        curve = result.curve("added")
        index = RIDGE_GRID.index(1.)
        self.assertAlmostEqual(result.base_model_edf, .5)
        self.assertAlmostEqual(curve.term_edf[index], 1. / 3.)
        self.assertAlmostEqual(curve.incremental_model_edf[index], 1. / 6.)
        self.assertAlmostEqual(curve.total_model_edf[index], 2. / 3.)

    def test_weighted_rank_deficient_base_matches_full_joint_influence(self):
        base = np.column_stack((np.ones(6), np.ones(6), np.arange(6.) / 5.))
        slot = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 2.], [2., -1.], [1., 3.]])
        weights = np.array([1., 0., 2., 1., 3., 1.])
        for penalties in (np.array([0., 0., .25]), np.zeros(3)):
            with self.subTest(penalties=tuple(penalties)):
                result = slot_capacity_curves(base_design=base, slot_designs={"added": slot},
                                              sample_weights=weights, base_penalties=penalties,
                                              base_rank_relative_tolerance=1e-12)
                self.assertEqual(result.base_numerical_rank, 2)
                self.assertEqual(result.ridge_penalties, RIDGE_GRID)
                curve = result.curve("added")
                expected = np.array([direct_traces(base, slot, weights, penalties, ridge)
                                     for ridge in RIDGE_GRID])
                observed = np.column_stack((curve.term_edf, curve.incremental_model_edf,
                                            curve.total_model_edf))
                np.testing.assert_allclose(observed, expected, atol=1e-9, rtol=1e-9)
                if not np.any(penalties):
                    np.testing.assert_allclose(curve.term_edf, curve.incremental_model_edf,
                                               atol=1e-12, rtol=1e-12)

    def test_actual_time_expanded_sequence_columns_use_the_same_weighted_objective(self):
        times = np.arange(7.) * .1
        sequences = []
        for name, shift in (("first", 0.), ("second", .03)):
            sequences.append(SequenceDesign(name, times, np.ones((7, 1)), {
                "linear": EventTerm(np.array([.01, .27]) + shift,
                                    np.array([[1.], [2.]]), basis, 2, (0., .4)),
                "added": EventTerm(np.array([.01, .27]) + shift,
                                   np.array([[2.], [-1.]]), basis, 2, (0., .4)),
            }))
        operator = JointSequenceDesign(sequences)
        expanded = operator @ np.eye(operator.shape[1])
        base, slot = expanded[:, :3], expanded[:, 3:]
        weights = np.array([1., 1., 0., 1., 2., 1., 1.] * 2)
        penalties = np.array([0., .1, .2])
        result = slot_capacity_curves(base_design=base, slot_designs={"added": slot},
                                      sample_weights=weights, base_penalties=penalties,
                                      base_rank_relative_tolerance=1e-12)
        curve = result.curve("added")
        expected = direct_traces(base, slot, weights, penalties, 1.)
        index = RIDGE_GRID.index(1.)
        np.testing.assert_allclose((curve.term_edf[index], curve.incremental_model_edf[index],
                                    curve.total_model_edf[index]), expected, atol=1e-10)

    def test_matching_keeps_the_exact_grid_five_percent_bound_and_direction(self):
        def curve(name, value):
            vector = (value,) * len(RIDGE_GRID)
            return SlotCapacityCurve(name, 30, (), vector, tuple(x / 2 for x in vector), vector)
        result = SlotCapacityResult(RIDGE_GRID, 0, 0, (), 1e-12, 0., 0., (
            curve("concept", 20.), curve("upper", 21.), curve("lower", 19.),
            curve("too_large", 21.00001)))
        options = dict(reference_slot="concept", comparison_kind="label", df_definition="term_edf")
        upper = admissible_capacity_pairs(result, comparison_slot="upper", **options)
        self.assertEqual(len(upper.admissible_pairs), 13 * 13)
        self.assertTrue(upper.operative_term_capacity)
        self.assertEqual(upper.unmatched_reference_penalty_indices, ())
        for name in ("lower", "too_large"):
            match = admissible_capacity_pairs(result, comparison_slot=name, **options)
            self.assertEqual(match.status, "underidentified")
            self.assertEqual(match.admissible_pairs, ())
        lower = admissible_capacity_pairs(result, comparison_slot="lower",
                                          **dict(options, comparison_kind="false_group"))
        self.assertEqual(len(lower.admissible_pairs), 13 * 13)
        diagnostic = admissible_capacity_pairs(result, comparison_slot="upper",
                                               **dict(options, df_definition="incremental_model_edf"))
        self.assertFalse(diagnostic.operative_term_capacity)

    def test_zero_slot_relative_capacity_and_invalid_training_support_refuse_qualification(self):
        options = dict(base_design=np.ones((3, 2)), sample_weights=np.ones(3),
                       base_penalties=np.zeros(2), base_rank_relative_tolerance=1e-12)
        result = slot_capacity_curves(slot_designs={"concept": np.empty((3, 0)),
                                                   "label": np.empty((3, 0))}, **options)
        self.assertEqual(result.base_numerical_rank, 1)
        matches = admissible_capacity_pairs(result, reference_slot="concept", comparison_slot="label",
                                            comparison_kind="label", df_definition="term_edf")
        self.assertEqual(matches.status, "underidentified")
        self.assertEqual(matches.zero_reference_penalty_indices, tuple(range(13)))
        for changes in (dict(sample_weights=np.zeros(3)), dict(base_design=np.ones((3, 2)) + 1j),
                        dict(base_rank_relative_tolerance=None)):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                slot_capacity_curves(slot_designs={"added": np.ones((3, 1))}, **dict(options, **changes))
        with self.assertRaises(ValueError):
            admissible_capacity_pairs(replace(result, ridge_penalties=(.1, 1.)),
                                      reference_slot="concept", comparison_slot="label",
                                      comparison_kind="label", df_definition="term_edf")


if __name__ == "__main__":
    unittest.main()
