"""Small full-draw known-answer checks of the registered inference rule."""
from pathlib import Path
from dataclasses import replace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import uncertainty_core as core
from prediction_core import ErrorSums, pooled_r2


def endpoint_statistic(offsets, scales=(1., 1.), phase="development"):
    """Known per-block error sums with equal-image SST, fixed two endpoints.

    This uses the production pooled-R² routine. Heterogeneity is deliberately
    small and nonzero, so analytic stable/reversal/equivalence answers are far
    from the fixed practical margins, rather than relying on a lucky seed.
    """
    blocks = 24 if phase == "development" else 12
    variation = (np.arange(4)[:, None] - 1.5) * .0008 + (np.arange(blocks)[None, :] - (blocks-1)/2) * .0001
    advantages = np.array(offsets)[None, None, :] + variation[:, :, None] * np.array(scales)
    model_a_errors = .3 - advantages
    model_b_errors = np.full_like(model_a_errors, .3)

    def statistic(participants, concept_blocks):
        result = np.zeros(2)
        for subject in range(4):
            # Twenty images per block; multiplicity retains all of each block.
            a = np.sum(model_a_errors[subject] * concept_blocks[:, None], axis=0) * 20
            b = np.sum(model_b_errors[subject] * concept_blocks[:, None], axis=0) * 20
            denominator = np.full(2, 20 * int(concept_blocks.sum()))
            score_a = pooled_r2([ErrorSums(a, denominator, int(denominator[0]))])
            score_b = pooled_r2([ErrorSums(b, denominator, int(denominator[0]))])
            result += participants[subject] * (score_a - score_b) / 4
        return result
    return statistic


class CrossedBootstrapTests(unittest.TestCase):
    def test_stable_positive_and_negative_have_margin_crossing_bounds(self):
        for offsets, expected in (((.08, .09), "stable_positive"), ((-.08, -.09), "stable_negative")):
            result = core.crossed_bounds(endpoint_statistic(offsets), ("before", "after"),
                                         ("raw_R2", "raw_R2"), "development")
            np.testing.assert_allclose(result.estimate, offsets, atol=1e-14)
            self.assertEqual(core.relation_decision(result, [("before", "after")]), expected)
            self.assertTrue((result.standard_error > 0).all())

    def test_genuine_reversal_needs_both_simultaneous_endpoints(self):
        result = core.crossed_bounds(endpoint_statistic((.08, -.09)), ("before", "after"),
                                     ("raw_R2", "raw_R2"), "development")
        self.assertEqual(core.relation_decision(result, [("before", "after")]), "genuine_reversal")
        self.assertEqual(core.relation_decision(result, []), "underidentified")

    def test_null_is_equivalence_only_with_narrow_bounds(self):
        result = core.crossed_bounds(endpoint_statistic((0., 0.), phase="audit"), ("before", "after"),
                                     ("raw_R2", "raw_R2"), "audit")
        self.assertEqual(core.relation_decision(result, [("before", "after")]), "practical_equivalence")
        self.assertTrue(core.audit_precision_adequate(result))
        uncertain = core.SimultaneousBounds(result.component_ids, result.score_scales,
            core.fixed([.001, -.001]), core.fixed([.1, .1]), 2., core.fixed([-.199, -.201]),
            core.fixed([.201, .199]), result.margins, "audit")
        self.assertEqual(core.relation_decision(uncertain, [("before", "after")]), "underidentified")
        self.assertFalse(core.audit_precision_adequate(uncertain))

    def test_crossed_order_and_original_centering_match_independent_reference(self):
        observed_weights = []
        def nonlinear(subjects, blocks):
            observed_weights.append((subjects.copy(), blocks.copy()))
            return [float((subjects @ np.arange(4) / 4) ** 2 + (blocks @ np.arange(12) / 12) ** 2)]
        got = core.crossed_bounds(nonlinear, ("nonlinear",), ("raw_R2",), "audit")
        generator = np.random.Generator(np.random.PCG64(1729))
        reference = []
        for i in range(10_000):
            subject_indices = generator.integers(0, 4, size=4)
            block_indices = generator.integers(0, 12, size=12)
            if i == 0:
                np.testing.assert_array_equal(observed_weights[1][0], np.bincount(subject_indices, minlength=4))
                np.testing.assert_array_equal(observed_weights[1][1], np.bincount(block_indices, minlength=12))
            reference.append(subject_indices.mean() ** 2 + block_indices.mean() ** 2)
        reference = np.asarray(reference)
        original = 1.5 ** 2 + 5.5 ** 2
        sd = np.sqrt(np.sum((reference - reference.mean()) ** 2) / 9999)
        expected_critical = sorted(abs((reference - original) / sd))[9500]
        self.assertAlmostEqual(got.standard_error[0], sd, places=12)
        self.assertAlmostEqual(got.critical_value, expected_critical, places=12)
        self.assertGreater(abs(reference.mean() - original), .1)

    def test_no_invalid_draw_redraw_or_constant_decimal_SE(self):
        calls = [0]
        def bad(subjects, blocks):
            calls[0] += 1
            return [np.nan if calls[0] == 2 else .1]
        with self.assertRaisesRegex(core.UncertaintyError, "undefined_statistic_no_redraw"):
            core.crossed_bounds(bad, ("a",), ("raw_R2",), "development")
        self.assertEqual(calls[0], 2)
        with self.assertRaisesRegex(core.UncertaintyError, "degenerate_bootstrap_standard_error"):
            core.crossed_bounds(lambda s,b: [.1], ("a",), ("raw_R2",), "development")

    def test_only_registered_mixed_scale_family_and_separate_margins(self):
        mixed = core.crossed_bounds(endpoint_statistic((.08, -.09), (1., 2.)),
                                    ("raw", "normalized"), ("raw_R2", "ceiling_normalized_R2"), "development")
        np.testing.assert_array_equal(mixed.margins, [.01, .05])
        self.assertEqual(core.relation_decision(mixed, [("raw", "normalized")]), "genuine_reversal")
        with self.assertRaisesRegex(core.UncertaintyError, "unregistered_mixed_scale_family"):
            core.crossed_bounds(lambda s,b: [1., 2.], ("a", "b"), ("raw_R2", "linear_CKA"), "development")

    def test_explanatory_components_share_bounds_but_are_not_primary_endpoints(self):
        primary = endpoint_statistic((.08, .09))
        def with_explanation(subjects, blocks):
            effects = primary(subjects, blocks)
            return np.r_[effects, -effects[0]]
        bounds = core.crossed_bounds(with_explanation, ("before", "after", "high_low"),
            ("raw_R2", "raw_R2", "beta_high_minus_low_gain"), "development",
            primary_endpoint_ids=("before", "after"))
        self.assertEqual(core.relation_decision(bounds, [("before", "after")]), "stable_positive")
        self.assertLess(bounds.upper[2], -.01)
        self.assertEqual(len(bounds.standard_error), 3)
        with self.assertRaisesRegex(core.UncertaintyError, "isolated_edge_identity_invalid"):
            core.relation_decision(bounds, [("before", "high_low")])
        all_primary = replace(bounds, primary_endpoint_ids=bounds.component_ids)
        self.assertEqual(core.relation_decision(all_primary, [("before", "after")]), "underidentified")


if __name__ == "__main__":
    unittest.main()
