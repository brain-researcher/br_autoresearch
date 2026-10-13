#!/usr/bin/env python3
"""Focused sealed-scope tests for the EP10 v3 M2 actual-K rank gate."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


STAGE0 = Path(__file__).resolve().parents[1]
GATE_PATH = STAGE0 / "tools" / "gate_v3_m2_actual_k.py"
SPEC = importlib.util.spec_from_file_location("ep10_gate_v3_m2_actual_k_tested", GATE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import M2 actual-K gate: {GATE_PATH}")
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def _cells() -> tuple[gate.m1.Cell, ...]:
    result: list[gate.m1.Cell] = []
    for index in range(gate.EXPECTED_ROWS):
        group = gate.m1.GROUP_ORDER[index % gate.EXPECTED_GROUPS]
        result.append(
            gate.m1.Cell(
                identifier=str(100000 + index),
                group=group,
                layer="MOp5",
                hemisphere="left",
                soma_um=(1.0, 2.0, 3.0),
                soma_distance_mm=np.linspace(0.1, 1.6, 16),
                k=index % 17,
                observed_index=0,
                segment_unmapped=index % 2,
            )
        )
    return tuple(result)


def _diagnostic(rank: int = gate.EXPECTED_RANK) -> dict[str, object]:
    return {
        "rank_at_tau_over_10_tau_ten_tau": {
            "tau_over_10": rank,
            "tau": rank,
            "ten_tau": rank,
        },
        "rank_stable": True,
        "rank": rank,
        "expected_rank": gate.EXPECTED_RANK,
        "expected_rank_match": rank == gate.EXPECTED_RANK,
        "range_projector_symmetry_maximum_absolute_error": 1.0e-15,
        "range_projector_idempotence_maximum_absolute_error": 2.0e-15,
        "G_FH_projection_maximum_absolute_orthogonality_error": 3.0e-15,
    }


class ContractAndRouteTests(unittest.TestCase):
    def test_frozen_contract_selects_exact_12_fold_rank_104_gate(self) -> None:
        gate.validate_gate_contract(
            STAGE0 / "v3_m2_search_contract.yaml",
            STAGE0 / "v3_m2_numerical_falsifier_addendum_v1.yaml",
        )

    def test_cli_has_no_membership_score_m0_swc_or_role_route(self) -> None:
        options = {
            option
            for action in gate.build_parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {
                "-h",
                "--help",
                "--contract",
                "--addendum",
                "--inventory",
                "--core-report",
                "--interface-root",
                "--geometry-root",
                "--target-root",
                "--output-root",
            },
        )
        for forbidden in ("membership", "score", "m0", "swc", "final", "unassigned"):
            self.assertFalse(any(forbidden in option.lower() for option in options))

    def test_reused_projection_parser_materializes_only_id_and_k(self) -> None:
        source = inspect.getsource(gate.sealed_k._scan_actual_k_projection)
        self.assertNotIn("csv.reader", source)
        self.assertNotIn("DictReader", source)
        self.assertEqual(
            gate.sealed_k.PERMITTED_TARGET_FIELDS,
            ("provider_neuron_id", "K"),
        )


class InventoryAndFoldTests(unittest.TestCase):
    def test_inventory_exact_join_and_group_match(self) -> None:
        cells = _cells()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "inventory.tsv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(gate.INVENTORY_FIELDS)
                writer.writerows((cell.identifier, cell.group) for cell in cells)
            inventory = gate._read_inventory(path)
            gate._validate_interface_inventory(cells, inventory)
            changed = dict(inventory)
            changed[cells[0].identifier] = cells[1].group
            with self.assertRaises(gate.M2ActualKGateError):
                gate._validate_interface_inventory(cells, changed)

    def test_exactly_12_logo_folds_and_no_full_development_fold(self) -> None:
        cells = _cells()
        folds = gate._outer_training_folds(cells)
        self.assertEqual(len(folds), 12)
        self.assertEqual(len(set(folds)), 12)
        for held_out, indexes in zip(gate.m1.GROUP_ORDER, folds, strict=True):
            training_groups = {cells[index].group for index in indexes}
            self.assertNotIn(held_out, training_groups)
            self.assertEqual(training_groups, set(gate.m1.GROUP_ORDER) - {held_out})
            self.assertLess(len(indexes), gate.EXPECTED_ROWS)

    def test_fold_runner_derives_rank_with_engine_expected_rank_argument(self) -> None:
        cells = _cells()
        fake_geometry = mock.Mock()
        fake_geometry.uniform_moments.side_effect = lambda cell: object()
        fake_transform = SimpleNamespace(diagnostics=_diagnostic())
        with (
            mock.patch.object(gate.m1, "CandidateLibrary", return_value=object()),
            mock.patch.object(gate.m1, "GeometryEngine", return_value=fake_geometry),
            mock.patch.object(gate.m2, "PairFeatureLibrary", return_value=object()),
            mock.patch.object(
                gate.m2,
                "fit_residual_transform",
                return_value=(fake_transform, object()),
            ) as fit,
        ):
            report = gate._compute_rank_gate(cells, object())
        self.assertEqual(fit.call_count, 12)
        for call in fit.call_args_list:
            self.assertEqual(call.kwargs, {"expected_rank": 104})
            self.assertLess(len(call.args[5]), gate.EXPECTED_ROWS)
        self.assertEqual(report["computed_rank_at_tau_unique"], [104])
        self.assertFalse(report["full_development_thirteenth_gate_added"])


class AggregateReportTests(unittest.TestCase):
    def test_aggregate_report_is_rank_only_and_contains_no_fold_rows(self) -> None:
        report = gate._aggregate_fold_diagnostics(
            [_diagnostic() for _ in range(gate.EXPECTED_FOLDS)]
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["training_folds_evaluated"], 12)
        self.assertFalse(report["fold_level_rows_emitted"])
        serialized = json.dumps(report, sort_keys=True).lower()
        for forbidden in (
            "provider_neuron_id",
            "cell_key\"",
            "conservative_group\"",
            "k_i_obs\"",
            "membership",
            "score",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_any_unstable_or_wrong_rank_fails_closed(self) -> None:
        cases = []
        wrong = _diagnostic(103)
        cases.append(wrong)
        unstable = _diagnostic()
        unstable["rank_at_tau_over_10_tau_ten_tau"] = {
            "tau_over_10": 105,
            "tau": 104,
            "ten_tau": 104,
        }
        cases.append(unstable)
        for bad in cases:
            diagnostics = [_diagnostic() for _ in range(12)]
            diagnostics[0] = bad
            with self.assertRaises(gate.M2ActualKGateError):
                gate._aggregate_fold_diagnostics(diagnostics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
