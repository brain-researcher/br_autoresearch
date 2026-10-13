#!/usr/bin/env python3
"""Target-free tests for the EP10 v3 optional-family diagnostic."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import v3_m2_optional_diagnostic_v1 as diagnostic  # noqa: E402


def _membership(k: int = 8) -> np.ndarray:
    result = np.zeros(16, dtype=float)
    result[:k] = 1.0
    return result


def _pair_indicator(membership: np.ndarray) -> np.ndarray:
    return np.asarray(
        [membership[left] * membership[right] for left, right in m2.PAIR_INDEX],
        dtype=float,
    )


def _diffuse_row(cell_key: str, group: str) -> diagnostic.PairResidualRow:
    membership = _membership(8)
    observed = _pair_indicator(membership)
    positive = 0.20
    negative = math.comb(8, 2) * positive / (120 - math.comb(8, 2))
    residual = np.where(observed == 1.0, positive, -negative)
    return diagnostic.PairResidualRow(
        cell_key,
        group,
        8,
        membership,
        observed - residual,
    )


class OptionalDiagnosticTests(unittest.TestCase):
    def test_diffuse_stable_pattern_activates_low_rank_gate(self) -> None:
        rows = tuple(
            _diffuse_row(f"cell_{group_index:02d}", group)
            for group_index, group in enumerate(m1.GROUP_ORDER)
        )
        result = diagnostic.diagnose_optional_families(rows)
        self.assertGreater(result.diagnostics.leading_energy, 0.79)
        self.assertGreater(result.diagnostics.leading_edge_participation_ratio, 36.0)
        self.assertEqual(result.diagnostics.median_leave_one_group_cosine, 1.0)
        self.assertEqual(result.diagnostics.leave_one_group_cosines_at_least_0_70, 12)
        self.assertEqual(result.leave_one_group_cosines, (1.0,) * 12)
        self.assertEqual(result.group_mean_residuals.shape, (12, 120))
        self.assertEqual(result.full_matrix.shape, (16, 16))
        self.assertFalse(result.full_matrix.flags.writeable)

    def test_localized_mass_transfer_has_half_spectral_energy(self) -> None:
        membership = _membership(2)
        observed = _pair_indicator(membership)
        probabilities = np.zeros(120, dtype=float)
        probabilities[m2.PAIR_TO_INDEX[(0, 2)]] = 1.0
        rows = tuple(
            diagnostic.PairResidualRow(
                f"cell_{group_index:02d}",
                group,
                2,
                membership,
                probabilities,
            )
            for group_index, group in enumerate(m1.GROUP_ORDER)
        )
        result = diagnostic.diagnose_optional_families(rows)
        self.assertAlmostEqual(result.diagnostics.leading_energy, 0.5, places=14)
        self.assertAlmostEqual(
            result.diagnostics.leading_edge_participation_ratio,
            25.0 / 9.0,
            places=14,
        )
        self.assertEqual(result.diagnostics.median_leave_one_group_cosine, 1.0)
        self.assertEqual(result.diagnostics.leave_one_group_cosines_at_least_0_70, 12)

    def test_exact_zero_residual_has_frozen_nontrigger_disposition(self) -> None:
        membership = _membership(8)
        observed = _pair_indicator(membership)
        rows = tuple(
            diagnostic.PairResidualRow(
                f"cell_{group_index:02d}",
                group,
                8,
                membership,
                observed,
            )
            for group_index, group in enumerate(m1.GROUP_ORDER)
        )
        result = diagnostic.diagnose_optional_families(rows)
        self.assertEqual(result.diagnostics.leading_energy, 0.0)
        self.assertEqual(result.diagnostics.leading_edge_participation_ratio, 1.0)
        self.assertEqual(result.diagnostics.median_leave_one_group_cosine, 0.0)
        self.assertEqual(result.diagnostics.leave_one_group_cosines_at_least_0_70, 0)
        np.testing.assert_array_equal(result.full_matrix, np.zeros((16, 16)))

    def test_equal_group_weighting_is_not_pooled_cell_weighting(self) -> None:
        membership = _membership(2)
        observed = _pair_indicator(membership)
        alternate = np.zeros(120, dtype=float)
        alternate[m2.PAIR_TO_INDEX[(0, 2)]] = 1.0
        rows: list[diagnostic.PairResidualRow] = []
        for group_index, group in enumerate(m1.GROUP_ORDER):
            repetitions = 7 if group_index == 0 else 1
            probabilities = alternate if group_index == 0 else observed
            for local_index in range(repetitions):
                rows.append(
                    diagnostic.PairResidualRow(
                        f"cell_{group_index:02d}_{local_index:02d}",
                        group,
                        2,
                        membership,
                        probabilities,
                    )
                )
        result = diagnostic.diagnose_optional_families(tuple(reversed(rows)))
        edge_01 = m2.PAIR_TO_INDEX[(0, 1)]
        self.assertEqual(result.group_mean_residuals[0, edge_01], 1.0)
        np.testing.assert_array_equal(result.group_mean_residuals[1:, edge_01], 0.0)
        self.assertAlmostEqual(result.full_matrix[0, 1], 1.0 / 12.0, places=15)
        self.assertAlmostEqual(result.full_matrix[0, 2], -1.0 / 12.0, places=15)

    def test_malformed_rows_fail_closed(self) -> None:
        membership = _membership(2)
        observed = _pair_indicator(membership)
        with self.assertRaisesRegex(diagnostic.OptionalDiagnosticError, "fixed K"):
            diagnostic.PairResidualRow(
                "bad_sum",
                m1.GROUP_ORDER[0],
                2,
                membership,
                observed * 0.5,
            )
        rows = tuple(
            diagnostic.PairResidualRow(
                "duplicate",
                group,
                2,
                membership,
                observed,
            )
            for group in m1.GROUP_ORDER
        )
        with self.assertRaisesRegex(diagnostic.OptionalDiagnosticError, "not unique"):
            diagnostic.diagnose_optional_families(rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
