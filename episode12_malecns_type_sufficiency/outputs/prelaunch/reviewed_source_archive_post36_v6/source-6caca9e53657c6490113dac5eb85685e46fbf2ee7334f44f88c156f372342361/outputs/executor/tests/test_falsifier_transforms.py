from __future__ import annotations

import unittest

import numpy as np

from ep12_executor.falsifier_transforms import (
    FalsifierTransformError,
    ablate_covariate_columns,
    binary_edge_counts,
    margin_preserving_switch_randomization,
    pool_partner_columns,
    shuffle_partner_identities_on_side,
    shuffle_side_membership,
)


class FalsifierTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counts = np.asarray(
            [
                [4, 0, 2, 1],
                [0, 5, 1, 1],
                [3, 1, 0, 2],
                [1, 2, 4, 0],
            ],
            dtype=np.int64,
        )

    def test_integer_switch_preserves_exact_margins_and_replays(self) -> None:
        first, receipt = margin_preserving_switch_randomization(
            self.counts, ["x", "x", "x", "x"], seed=17
        )
        second, second_receipt = margin_preserving_switch_randomization(
            self.counts, ["x", "x", "x", "x"], seed=17
        )
        np.testing.assert_array_equal(first.sum(0), self.counts.sum(0))
        np.testing.assert_array_equal(first.sum(1), self.counts.sum(1))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(receipt.output_sha256, second_receipt.output_sha256)
        self.assertGreater(receipt.details["completed_switches"], 0)

    def test_partner_shuffle_retains_endpoint_columns_and_neuron_totals(self) -> None:
        result, receipt = shuffle_partner_identities_on_side(
            self.counts, ["L", "L", "R", "R"], [0, 1, 2], shuffled_side="R", seed=9
        )
        np.testing.assert_array_equal(result.sum(1), self.counts.sum(1))
        np.testing.assert_array_equal(result[:, 3], self.counts[:, 3])
        np.testing.assert_array_equal(result[:2], self.counts[:2])
        self.assertTrue(receipt.details["endpoint_columns_unchanged"])

    def test_side_shuffle_preserves_counts_in_each_type(self) -> None:
        result, receipt = shuffle_side_membership(
            ["L", "L", "R", "R", "L", "R"],
            ["a", "a", "a", "a", "b", "b"],
            seed=3,
        )
        self.assertEqual(receipt["stratum_side_counts_before"], receipt["stratum_side_counts_after"])
        self.assertEqual(sorted(result.tolist()), ["L", "L", "L", "R", "R", "R"])

    def test_binary_transform_preserves_support_not_strength(self) -> None:
        result, receipt = binary_edge_counts(self.counts)
        np.testing.assert_array_equal(result > 0, self.counts > 0)
        self.assertTrue(receipt.details["positive_support_preserved"])
        self.assertFalse(receipt.total_mass_preserved)

    def test_partner_pooling_retains_all_mass(self) -> None:
        result, receipt = pool_partner_columns(self.counts, [0, 2], pool_column=3)
        np.testing.assert_array_equal(result.sum(1), self.counts.sum(1))
        self.assertEqual(int(result[:, [0, 2]].sum()), 0)
        self.assertEqual(int(result[:, 3].sum()), int(self.counts[:, [0, 2, 3]].sum()))
        self.assertTrue(receipt.total_mass_preserved)

    def test_common_covariate_ablation(self) -> None:
        covariates = np.arange(24, dtype=float).reshape(4, 6)
        result, receipt = ablate_covariate_columns(
            covariates, [1, 4], block_id="anatomy"
        )
        np.testing.assert_array_equal(result[:, [1, 4]], 0.0)
        np.testing.assert_array_equal(result[:, [0, 2, 3, 5]], covariates[:, [0, 2, 3, 5]])
        self.assertEqual(receipt["applied_to_models"], ["T", "U", "M"])

    def test_invalid_switch_fails_closed(self) -> None:
        with self.assertRaises(FalsifierTransformError):
            margin_preserving_switch_randomization(
                np.asarray([[1, 0], [2, 0]], dtype=np.int64),
                ["x", "x"],
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
