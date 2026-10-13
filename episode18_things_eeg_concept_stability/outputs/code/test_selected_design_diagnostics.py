"""Focused exact-operator/acceptance regressions; coordinated compute only."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import selected_design_diagnostics as diagnostics
from slot_capacity import RIDGE_GRID, slot_capacity_curves


def fixture():
    times = np.arange(8., dtype=np.float64) / 7.
    base = np.column_stack((np.ones(8), np.ones(8), times))
    slot = np.array([[1., 0., 2.], [0., 1., -1.], [1., 1., .5], [-1., 2., 0.],
                     [2., -1., 3.], [1., 3., -2.], [.2, -2., 1.], [3., .1, -1.]])
    weights = np.array([1., 0., 2., 1., 3., 1., 2., 1.])
    penalties = np.array([0., 0., .25])
    return base, slot, weights, penalties


def prepare(base, weights, penalties, **changes):
    options = dict(recipe_id=diagnostics.RECIPE_ID, base_design=base, sample_weights=weights,
                   base_penalties=penalties, base_rank_relative_tolerance=1e-12,
                   array_byte_cap=8*1024**2, projection_chunk_rows=2)
    return diagnostics.prepare_selected_base(**dict(options, **changes))


def direct_joint(base, slot, weights, penalties, ridge):
    """Independent explicit full augmented least-squares smoother, including base refit."""
    root = np.sqrt(weights)[:, None]
    zbase, zslot = root * base, root * slot
    design = np.column_stack((zbase, zslot))
    full_penalty = np.concatenate((penalties, np.full(slot.shape[1], ridge)))
    augmented = np.vstack((design, np.diag(np.sqrt(full_penalty))))
    coefficient_influence = np.linalg.pinv(augmented, rcond=1e-12)[:, :len(base)]
    smoother = design @ coefficient_influence
    term = zslot @ coefficient_influence[base.shape[1]:]
    return smoother, float(np.trace(term))


class SelectedDesignDiagnosticsTests(unittest.TestCase):
    def test_shared_factorization_matches_explicit_joint_smoother_and_existing_edf(self):
        base, slot, weights, penalties = fixture()
        shared = prepare(base, weights, penalties)
        public = slot_capacity_curves(base_design=base, slot_designs={"true_concept": slot},
            sample_weights=weights, base_penalties=penalties, base_rank_relative_tolerance=1e-12)
        selected = diagnostics.evaluate_selected_slot(shared, slot_name="true_concept",
            slot_design=slot, selected_penalty_index=6)
        self.assertEqual(shared.factors.numerical_rank, 2)
        self.assertEqual(len(selected.curve.term_edf), 13)
        np.testing.assert_allclose(selected.curve.term_edf, public.curve("true_concept").term_edf,
                                   atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(selected.curve.total_model_edf, public.curve("true_concept").total_model_edf,
                                   atol=1e-12, rtol=1e-12)
        for index in (0, 6, 12):
            result = selected if index == 6 else diagnostics.evaluate_selected_slot(shared,
                slot_name="true_concept", slot_design=slot, selected_penalty_index=index)
            smoother, term = direct_joint(base, slot, weights, penalties, RIDGE_GRID[index])
            np.testing.assert_allclose(result.joint_leverage, np.diag(smoother), atol=1e-10, rtol=1e-10)
            self.assertAlmostEqual(result.selected_term_edf, term, places=9)
            self.assertAlmostEqual(sum(result.joint_leverage), result.curve.total_model_edf[index], places=10)
            self.assertEqual(result.joint_leverage[1], 0.)
        self.assertGreater(abs(selected.curve.term_edf[6] - selected.curve.incremental_model_edf[6]), 1e-6)
        # Independent Schur Gram via the complete augmented base inverse.
        zbase, zslot = np.sqrt(weights)[:, None]*base, np.sqrt(weights)[:, None]*slot
        augmented = np.vstack((zbase, np.diag(np.sqrt(penalties))))
        hbase = zbase @ np.linalg.pinv(augmented, rcond=1e-12)[:, :len(base)]
        gram = zslot.T @ (np.eye(len(base)) - hbase) @ zslot
        expected = np.linalg.eigvalsh(gram)[::-1]
        np.testing.assert_allclose(selected.conditional_gram_eigenvalues, expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.square(selected.design_singular_values), expected, atol=1e-10, rtol=1e-10)
        with self.assertRaises(ValueError):
            selected.joint_leverage[0] = 0.
        # More columns than root rows retain structural zeros, not truncated directions.
        small = prepare(np.empty((3, 0)), np.ones(3), np.empty(0))
        wide = diagnostics.evaluate_selected_slot(small, slot_name="true_concept",
            slot_design=np.column_stack((np.eye(3), np.zeros((3, 2)))), selected_penalty_index=6)
        self.assertEqual(wide.design_singular_values, (1., 1., 1., 0., 0.))

    def test_every_distribution_bound_zero_rule_and_exact_empirical_wasserstein(self):
        ref = np.array([.1, 1., 3., 5., 9.])
        other = np.array([9.01, 1.01, .11, 3.01, 5.01])
        result = diagnostics.compare_design_distribution(ref, other)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.wasserstein_1, .01)
        self.assertAlmostEqual(result.reference_sample_sd, np.std(ref, ddof=1))
        np.testing.assert_allclose(result.reference_quantiles, np.quantile(ref, diagnostics.QUANTILES, method="linear"))
        # Generic standardized bounds can fail while relative quantiles pass.
        standardized_only_failure = diagnostics.compare_design_distribution([99., 100., 101.], [99.2, 100.2, 101.2])
        self.assertFalse(standardized_only_failure.standardized_pass)
        self.assertTrue(standardized_only_failure.relative_quantile_pass)
        # Relative quantiles can fail while every standardized discrepancy passes.
        near_zero = np.array([0., .001, 1., 2., 3.])
        relative_only_failure = diagnostics.compare_design_distribution(near_zero, near_zero + .005)
        self.assertTrue(relative_only_failure.standardized_pass)
        self.assertFalse(relative_only_failure.relative_quantile_pass)
        self.assertTrue(diagnostics.compare_design_distribution([0., 0.], [0., .5e-12]).passed)
        self.assertFalse(diagnostics.compare_design_distribution([0., 0.], [0., 2e-12]).passed)
        constant = diagnostics.compare_design_distribution([1., 1.], [1. + 1e-10]*2)
        self.assertTrue(constant.relative_quantile_pass)
        self.assertFalse(constant.zero_scale_exact_equality)
        with self.assertRaises(ValueError):
            diagnostics.compare_design_distribution([1.], [1.])
        with self.assertRaises(ValueError):
            diagnostics.compare_design_distribution([0., 1.], [np.nan, 1.])

    def test_exact_eight_slots_six_folds_frozen_bank_and_no_penalty_fallback(self):
        base, slot, weights, penalties = fixture()
        shared = prepare(base, weights, penalties)
        true = diagnostics.evaluate_selected_slot(shared, slot_name="true_concept",
            slot_design=slot, selected_penalty_index=6)
        false = {name: diagnostics.evaluate_selected_slot(shared, slot_name=name,
            slot_design=slot[:, ::-1], selected_penalty_index=6) for name in diagnostics.FALSE_SLOTS}
        identities = tuple(f"{index:064x}" for index in range(8))
        options = dict(outer_index=0, true_slot=true, false_slots=false, frozen_partition_fingerprints=identities)
        fold = diagnostics.compare_selected_outer_fold(**options)
        self.assertTrue(fold.passed)
        self.assertEqual(fold.common_concept_penalty_index, 6)
        complete = tuple(replace(fold, outer_index=index) for index in range(6))
        result = diagnostics.assemble_selected_outer_folds(complete)
        self.assertEqual(result["status"], "selected_design_checks_pass")
        self.assertFalse(result["false_bank_fully_qualified"])
        with self.assertRaises(ValueError):
            diagnostics.assemble_selected_outer_folds(complete[:-1])
        with self.assertRaises(ValueError):
            diagnostics.assemble_selected_outer_folds(complete[:-1] + (complete[0],))
        with self.assertRaises(ValueError):
            diagnostics.assemble_selected_outer_folds(complete[:-1] + (replace(complete[-1],
                frozen_partition_fingerprints=identities[::-1]),))
        with self.assertRaises(ValueError):
            diagnostics.compare_selected_outer_fold(**dict(options, false_slots=dict(list(false.items())[:-1])))
        different_base = prepare(base, weights, penalties)
        foreign = diagnostics.evaluate_selected_slot(different_base, slot_name="false_group_1",
            slot_design=slot, selected_penalty_index=6)
        with self.assertRaises(ValueError):
            diagnostics.compare_selected_outer_fold(**dict(options, false_slots=dict(false, false_group_1=foreign)))
        # An admissible alternative exists at 6, but the already selected 12 must fail.
        mismatched = diagnostics.evaluate_selected_slot(shared, slot_name="false_group_1",
            slot_design=slot, selected_penalty_index=12)
        failed = diagnostics.compare_selected_outer_fold(**dict(options, false_slots=dict(false, false_group_1=mismatched)))
        self.assertFalse(failed.passed)
        self.assertFalse(failed.false_matches[0].term_edf_pass)
        self.assertEqual(failed.false_matches[0].comparison_penalty_index, 12)
        self.assertEqual(diagnostics.assemble_selected_outer_folds(complete[:-1] + (replace(failed, outer_index=5),))
                         ["status"], "selected_design_match_failure")

    def test_resource_admission_precedes_dense_factorization_without_approximating(self):
        base, slot, weights, penalties = fixture()
        with patch.object(diagnostics, "_factor_base") as factor:
            with self.assertRaises(MemoryError):
                prepare(base, weights, penalties, array_byte_cap=1)
            factor.assert_not_called()
        shared = prepare(base, weights, penalties)
        with patch.object(diagnostics, "_evaluate_slot") as factor:
            with self.assertRaises(MemoryError):
                diagnostics.evaluate_selected_slot(replace(shared, array_byte_cap=1),
                    slot_name="true_concept", slot_design=slot, selected_penalty_index=6)
            factor.assert_not_called()
        estimate = diagnostics.estimate_array_bytes(rows=500000, base_columns=10000,
            slot_columns=74160, projection_chunk_rows=64)
        self.assertGreater(estimate["required_array_bytes"], 256*1024**3)
        self.assertFalse(estimate["native_workspace_included"])
        # The bound includes the caller's base and the concurrently live
        # selected-factor/square intermediates when rows dominate columns.
        tall = diagnostics.estimate_array_bytes(rows=100000, base_columns=4,
            slot_columns=0, projection_chunk_rows=2)
        self.assertGreaterEqual(tall["base_factorization_bytes"], 8 * 7 * 100000 * 4)
        for changed in (dict(recipe_id="unselected_definition"), dict(projection_chunk_rows=0)):
            with self.assertRaises(ValueError):
                prepare(base, weights, penalties, **changed)


if __name__ == "__main__":
    unittest.main()
