"""Small source-free membership/weight fixtures for EP17 support construction."""

from collections import Counter
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import support_core as core


def metadata(counts=(35,) * 8, participant="sub-01", roi="left"):
    return tuple(core.StructuralColumn(participant, roi, 1000 + 1000 * index + 3 * within,
                                       spatial_bin, core.ROI_LABELS[roi])
                 for index, (spatial_bin, count) in enumerate(zip(core.BINS, counts))
                 for within in range(count))


def ids(columns):
    return tuple(row.column_id for row in columns)


def zero_reliability(columns):
    return {row.column_id: 0.0 for row in columns}


class SupportCoreTests(unittest.TestCase):
    def test_exact_hash_namespace_and_every_bank_quota_with_original_ids(self):
        expected = hashlib.sha256(b"ep17-anatomical-bank-v1|17|sub-01|left|BIN01|BANK01|1037").digest()
        self.assertEqual(core._digest("sub-01", "left", "BIN01", "BANK01", 1037), expected)
        columns = metadata(tuple(range(35, 43)))
        bank = core.build_anatomical_bank(columns, ids(columns))
        lookup = {row.column_id: row.spatial_bin for row in columns}
        self.assertEqual(bank.column_ids, tuple(sorted(ids(columns))))
        self.assertEqual(len(bank.banks), 32)
        for selected in bank.banks:
            self.assertEqual(len(selected), 256)
            self.assertEqual(len(set(selected)), 256)
            self.assertEqual(selected, tuple(sorted(selected)))
            self.assertEqual(Counter(lookup[value] for value in selected), dict.fromkeys(core.BINS, 32))
            self.assertTrue(set(selected).issubset(ids(columns)))
        self.assertEqual(sum(bank.inclusion_counts), 32 * 256)
        independent_counts = Counter(value for selected in bank.banks for value in selected)
        self.assertEqual(bank.inclusion_counts, tuple(independent_counts[value] for value in bank.column_ids))
        self.assertEqual(bank.mean_weights, tuple(independent_counts[value] / 8192 for value in bank.column_ids))
        for spatial_bin in core.BINS:
            self.assertEqual(sum(weight for value, weight in zip(bank.column_ids, bank.mean_weights)
                                 if lookup[value] == spatial_bin), 1 / 8)
        self.assertEqual(sum(bank.mean_weights), 1)
        self.assertAlmostEqual(sum(bank.a_all_weights), 1, places=13)
        self.assertEqual(bank.member_weight, 1 / 256)

    def test_input_order_and_ID_gaps_do_not_change_either_support(self):
        columns = metadata()
        eligible = ids(columns)
        reliability = {row.column_id: (index % 35) / 35 for index, row in enumerate(columns)}
        first = core.build_anatomical_bank(columns, eligible)
        second = core.build_anatomical_bank(columns[::-1], eligible[::-1])
        self.assertEqual(first, second)
        self.assertEqual(core.build_reliable_support(columns, eligible, reliability),
                         core.build_reliable_support(columns[::-1], eligible[::-1], dict(reversed(tuple(reliability.items())))))

    def test_reliable_ties_keep_lowest_original_columns_within_each_bin(self):
        columns = metadata()
        reliability = dict.fromkeys(ids(columns), -0.25)
        support = core.build_reliable_support(columns, ids(columns), reliability)
        expected = tuple(sorted(row.column_id for index, row in enumerate(columns) if index % 35 < 32))
        self.assertEqual(support.selected_ids, expected)
        self.assertEqual(support.finite_eligible_counts_by_bin, (35,) * 8)
        self.assertEqual(sum(support.weights), 1)
        self.assertEqual(sum(weight != 0 for weight in support.weights), 256)

    def test_ranking_uses_unclipped_Pearson_and_anatomical_bank_ignores_reliability(self):
        columns = metadata()
        eligible = ids(columns)
        bank = core.build_anatomical_bank(columns, eligible)
        # All values would tie if incorrectly Fisher-boundary clipped at1-1e-7.
        reliability = {row.column_id: 1 - (35 - index % 35) * 1e-10 for index, row in enumerate(columns)}
        support = core.build_reliable_support(columns, eligible, reliability)
        expected = tuple(sorted(row.column_id for index, row in enumerate(columns) if index % 35 >= 3))
        self.assertEqual(support.selected_ids, expected)
        unmeasurable = bank.banks[0][0]
        reliability[unmeasurable] = float("nan")
        changed = core.build_reliable_support(columns, eligible, reliability)
        self.assertNotIn(unmeasurable, changed.selected_ids)
        self.assertIn(unmeasurable, bank.banks[0])
        self.assertEqual(core.build_anatomical_bank(columns, eligible), bank)

    def test_ineligible_A_all_voxels_never_enter_any_bank_or_reliable_support(self):
        columns = metadata()
        excluded = {row.column_id for index, row in enumerate(columns) if index % 35 == 0}
        eligible = tuple(value for value in ids(columns) if value not in excluded)
        reliability = zero_reliability(columns)
        for value in excluded:
            reliability[value] = 1.0
        bank = core.build_anatomical_bank(columns, eligible)
        reliable = core.build_reliable_support(columns, eligible, reliability)
        self.assertTrue(all(not (set(subset) & excluded) for subset in bank.banks))
        self.assertFalse(set(reliable.selected_ids) & excluded)
        self.assertTrue(all(count == 0 for value, count in zip(bank.column_ids, bank.inclusion_counts) if value in excluded))

    def test_insufficient_A_all_or_finite_reliability_refuses_without_lowering_quota(self):
        columns = metadata()
        too_few = tuple(row.column_id for index, row in enumerate(columns) if not (index < 4))
        with self.assertRaisesRegex(core.SupportError, "insufficient_A_all_bin_quota"):
            core.build_anatomical_bank(columns, too_few)
        reliability = zero_reliability(columns)
        for value, invalid in zip(ids(columns)[:4], (float("nan"), float("inf"), float("-inf"), float("nan"))):
            reliability[value] = invalid
        with self.assertRaisesRegex(core.SupportError, "insufficient_finite_D_bin_quota"):
            core.build_reliable_support(columns, ids(columns), reliability)
        # The anatomical size-control support still exists independently.
        self.assertEqual(len(core.build_anatomical_bank(columns, ids(columns)).banks), 32)

    def test_identity_group_label_bin_and_reliability_join_refusals(self):
        columns = metadata()
        eligible = ids(columns)
        changed = (
            columns + columns[:1],
            (replace(columns[0], participant="sub-02"),) + columns[1:],
            (replace(columns[0], roi="right"),) + columns[1:],
            (replace(columns[0], atlas_label=2007),) + columns[1:],
            (replace(columns[0], column_id=227214),) + columns[1:],
            (replace(columns[0], spatial_bin="BIN00"),) + columns[1:],
            (replace(columns[0], column_id=True),) + columns[1:],
        )
        for example in changed:
            with self.subTest(first=example[0]), self.assertRaises(core.SupportError):
                core.build_anatomical_bank(example, eligible)
        for wrong in (eligible + eligible[:1], eligible + (999,)):
            with self.assertRaises(core.SupportError):
                core.build_anatomical_bank(columns, wrong)
        reliability = zero_reliability(columns)
        del reliability[eligible[0]]
        with self.assertRaisesRegex(core.SupportError, "reliability_original_ID_mismatch"):
            core.build_reliable_support(columns, eligible, reliability)
        reliability[eligible[0]] = 1.001
        with self.assertRaisesRegex(core.SupportError, "D_reliability_outside_Pearson_range"):
            core.build_reliable_support(columns, eligible, reliability)

    def test_digest_ties_arithmetic_weights_overlap_and_no_bank_reroll(self):
        columns = metadata()
        eligible = ids(columns)
        with patch.object(core, "_digest", return_value=b"\0" * 32):
            bank = core.build_anatomical_bank(columns, eligible)
        expected = tuple(sorted(row.column_id for index, row in enumerate(columns) if index % 35 < 32))
        self.assertEqual(bank.banks, (expected,) * 32)
        self.assertEqual(bank.inclusion_counts, tuple(32 if value in expected else 0 for value in bank.column_ids))
        self.assertEqual(bank.mean_weights, tuple(1 / 256 if value in expected else 0 for value in bank.column_ids))
        overlap = bank.overlap_summary()
        self.assertEqual(overlap["unique_bank_count"], 1)
        self.assertEqual(overlap["union_column_count"], 256)
        self.assertTrue(all(value == 256 for row in overlap["pairwise_intersection_counts"] for value in row))
        self.assertFalse(overlap["bank_members_are_independent_replicates"])
        reliability = {row.column_id: (index % 35) / 35 for index, row in enumerate(columns)}
        reliable = core.build_reliable_support(columns, eligible, reliability)
        comparison = core.compare_supports(bank, reliable)
        self.assertEqual(comparison["bank_reliable_intersection_counts"], (232,) * 32)
        self.assertEqual(comparison["bank_reliable_jaccard"], (232 / 280,) * 32)
        self.assertEqual(sum(comparison["reliable_minus_anatomical_mean_weights"]), 0)
        self.assertEqual(comparison["independent_trial_count_added"], 0)
        with self.assertRaisesRegex(core.SupportError, "support_group_or_A_all_mismatch"):
            core.compare_supports(bank, replace(reliable, participant="sub-02"))


if __name__ == "__main__":
    unittest.main()
