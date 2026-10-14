"""Three affected-backend crosschecks; coordinator-owned compute only.

Reuse the existing capacity/selected-design fixtures and dense mathematical
oracle. No empirical inputs, scientific simulations or new search are used.
"""
import unittest
from unittest.mock import patch

import numpy as np

from continuous_sequence_model import EventTerm, JointSequenceDesign, SequenceDesign
from slot_capacity import RIDGE_GRID, slot_capacity_curves
import streamed_slot_capacity as streamed
import test_selected_design_diagnostics as prior
import test_slot_capacity as existing


def impulse(lags):
    return np.ones((len(lags), 1))


def from_arrays(base, slot):
    """Use actual sequence operators to represent the prior dense fixture."""
    bases, full = [], []
    cut = len(base)//2
    for index, rows in enumerate((slice(0, cut), slice(cut, len(base)))):
        values, added = base[rows], slot[rows]
        times = np.arange(len(values), dtype=float) / 250
        terms = {"slot": EventTerm(times, added, impulse, 1, (0., 0.))}
        # A zero-width base can be a prefix of the same original operator.
        if base.shape[1]:
            bases.append(SequenceDesign(str(index), times, values, {}))
        full.append(SequenceDesign(str(index), times, values, terms))
    combined = JointSequenceDesign(full)
    baseline = JointSequenceDesign(bases) if bases else combined
    return (streamed.SequenceRows(baseline, 0, base.shape[1]),
            streamed.SequenceRows(combined, base.shape[1], base.shape[1]+slot.shape[1]))


def run(base, slot, weights, penalties, **changes):
    first, second = from_arrays(base, slot)
    options = dict(base_design=first, slot_designs={"added": second},
        sample_weights=weights, base_penalties=penalties, base_rank_relative_tolerance=1e-12,
        chunk_rows=2, qr_block_size=2, array_byte_cap=16*1024**2,
        leverage_penalty_indices={"added": (0, 6, 12)}, budget_check=lambda: None)
    return streamed.streamed_slot_capacity_curves(**dict(options, **changes))


class StreamedCapacityTests(unittest.TestCase):
    def test_actual_unrounded_events_row_blocks_match_complete_operator_and_boundaries(self):
        times = np.arange(7.) * .1
        sequences, baselines = [], []
        for name, shift in (("first", 0.), ("second", .03)):
            terms = {
                "linear": EventTerm(np.array([.01, .27])+shift,
                    np.array([[1.], [2.]]), existing.basis, 2, (0., .4)),
                "added": EventTerm(np.array([.01, .27])+shift,
                    np.array([[2.], [-1.]]), existing.basis, 2, (0., .4))}
            sequences.append(SequenceDesign(name, times, np.ones((7, 1)), terms))
            baselines.append(SequenceDesign(name, times, np.ones((7, 1)), {"linear": terms["linear"]}))
        operator = JointSequenceDesign(sequences)
        dense = operator @ np.eye(operator.shape[1])
        blocks = tuple(streamed.SequenceRows(operator, 0, 5).blocks(chunk_rows=3, budget_check=lambda: None))
        self.assertEqual(tuple((lo, hi) for lo, hi, _ in blocks),
                         ((0, 3), (3, 6), (6, 7), (7, 10), (10, 13), (13, 14)))
        np.testing.assert_allclose(np.concatenate([value for _, _, value in blocks]), dense, rtol=0, atol=1e-14)
        for first, last in ((0, 1), (1, 3), (3, 5), (2, 4)):
            actual = np.concatenate([value for _, _, value in
                streamed.SequenceRows(operator, first, last).blocks(chunk_rows=2, budget_check=lambda: None)])
            np.testing.assert_allclose(actual, dense[:, first:last], atol=1e-14, rtol=0)
        weights = np.asarray([1., 1., 0., 1., 2., 1., 1.] * 2)
        result, leverage = streamed.streamed_slot_capacity_curves(
            base_design=streamed.SequenceRows(JointSequenceDesign(baselines), 0, 3),
            slot_designs={"added": streamed.SequenceRows(operator, 3, 5)},
            sample_weights=weights, base_penalties=np.array([0., .1, .2]),
            base_rank_relative_tolerance=1e-12, chunk_rows=3, qr_block_size=2,
            array_byte_cap=16*1024**2, leverage_penalty_indices={"added": (6,)}, budget_check=lambda: None)
        expected = existing.direct_traces(dense[:, :3], dense[:, 3:], weights, np.array([0., .1, .2]), 1.)
        curve = result.curve("added")
        np.testing.assert_allclose([curve.term_edf[6], curve.incremental_model_edf[6],
                                   curve.total_model_edf[6]], expected, atol=1e-10)
        self.assertAlmostEqual(float(leverage["added"][6].sum()), curve.total_model_edf[6], places=10)
        # An altered second physical sequence is not a compatible shared row axis.
        # Construct the changed clock with original EventTerm objects, not prepared terms.
        shifted = SequenceDesign("second", times+.001, np.ones((7, 1)), terms)
        with self.assertRaises(ValueError):
            streamed._same_rows_and_base(streamed.SequenceRows(JointSequenceDesign(baselines), 0, 3),
                streamed.SequenceRows(JointSequenceDesign((sequences[0], shifted)), 3, 5))

    def test_all_penalties_full_spectrum_rank_deficient_base_and_joint_leverage_match_dense(self):
        base, slot, weights, original_penalties = prior.fixture()
        for penalties in (np.zeros(3), original_penalties, np.array([.5, .75, .25])):
            with self.subTest(penalties=tuple(penalties)):
                result, leverages = run(base, slot, weights, penalties)
                expected = slot_capacity_curves(base_design=base, slot_designs={"added": slot},
                    sample_weights=weights, base_penalties=penalties, base_rank_relative_tolerance=1e-12)
                self.assertEqual(result.base_numerical_rank, expected.base_numerical_rank)
                self.assertEqual(result.ridge_penalties, RIDGE_GRID)
                np.testing.assert_allclose(result.base_singular_values, expected.base_singular_values, atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(result.base_rank_threshold, expected.base_rank_threshold, places=14)
                self.assertAlmostEqual(result.base_model_edf, expected.base_model_edf, places=11)
                for field in ("schur_singular_values", "term_edf", "incremental_model_edf", "total_model_edf"):
                    np.testing.assert_allclose(getattr(result.curve("added"), field),
                        getattr(expected.curve("added"), field), atol=1e-10, rtol=1e-10)
                for index in (0, 6, 12):
                    smoother, term = prior.direct_joint(base, slot, weights, penalties, RIDGE_GRID[index])
                    np.testing.assert_allclose(leverages["added"][index], np.diag(smoother), atol=1e-9, rtol=1e-9)
                    self.assertAlmostEqual(result.curve("added").term_edf[index], term, places=8)
                    self.assertEqual(leverages["added"][index][1], 0.)
                    self.assertFalse(leverages["added"][index].flags.writeable)
        # Reuse the prior wider-than-root case: preserve original rectangular
        # spectrum length and expose only mathematically structural zero padding.
        empty_base = np.empty((3, 0))
        wide = np.column_stack((np.eye(3), np.zeros((3, 2))))
        first, second = from_arrays(empty_base, wide)
        prepared = streamed.prepare_streamed_base(base_design=first, sample_weights=np.ones(3),
            base_penalties=np.empty(0), base_rank_relative_tolerance=1e-12,
            chunk_rows=2, qr_block_size=2, array_byte_cap=16*1024**2,
            retained_leverage_vectors=1, budget_check=lambda: None)
        result = streamed.evaluate_streamed_slot(prepared, slot_name="wide", slot_design=second,
                                                 leverage_penalty_indices=(6,))
        self.assertEqual(len(result.curve.schur_singular_values), 3)
        np.testing.assert_allclose(result.design_singular_values, (1., 1., 1., 0., 0.), atol=1e-12)

    def test_admission_and_budget_fail_without_factorization_or_partial_results(self):
        base, slot, weights, penalties = prior.fixture()
        with patch.object(streamed.lapack, "dtpqrt", side_effect=AssertionError("factorization must not start")):
            with self.assertRaises(MemoryError):
                run(base, slot, weights, penalties, array_byte_cap=1)
            with self.assertRaises(ValueError):
                run(base, slot, weights, penalties, leverage_penalty_indices={"added": (6, 6)})
            with self.assertRaises(ValueError):
                run(base, slot, np.zeros(len(weights)), penalties)
            with self.assertRaises(FloatingPointError):
                run(np.full_like(base, 1e308), slot, np.full(len(weights), 100.), penalties)
        calls = []
        def stop():
            calls.append(True)
            raise TimeoutError("selected budget ended")
        with self.assertRaises(TimeoutError):
            run(base, slot, weights, penalties, budget_check=stop)
        self.assertEqual(len(calls), 1)
        # The estimate has square factors and row blocks, not retained n*q arrays.
        options = dict(base_columns=3, slot_columns=5, chunk_rows=2, qr_block_size=2,
                       retained_leverage_vectors=3)
        small = streamed.estimate_array_bytes(rows=100, **options)["required_array_bytes"]
        larger = streamed.estimate_array_bytes(rows=200, **options)["required_array_bytes"]
        self.assertEqual(larger-small, 8*100*(20+3))


if __name__ == "__main__":
    unittest.main()
