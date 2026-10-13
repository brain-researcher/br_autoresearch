#!/usr/bin/env python3
"""Focused sealed-scope tests for the EP10 v3 M1 actual-K-only gate."""

from __future__ import annotations

import copy
import csv
import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


GATE_PATH = Path(__file__).resolve().parents[1] / "tools" / "gate_v3_m1_actual_k.py"
SPEC = importlib.util.spec_from_file_location("ep10_gate_v3_m1_actual_k_tested", GATE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import actual-K gate: {GATE_PATH}")
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

TARGET_HEADER = [f"sealed_field_{index}" for index in range(gate.TARGET_FIELD_COUNT)]
TARGET_HEADER[gate.TARGET_IDENTIFIER_INDEX] = "provider_neuron_id"
TARGET_HEADER[gate.TARGET_K_INDEX] = "K"


def _target_row(identifier: str, k: str, sealed_value: str = "") -> list[str]:
    row = [""] * gate.TARGET_FIELD_COUNT
    row[gate.TARGET_IDENTIFIER_INDEX] = identifier
    row[gate.TARGET_K_INDEX] = k
    row[17] = sealed_value
    return row


def _cell(index: int) -> gate.m1.Cell:
    group = gate.m1.GROUP_ORDER[index % len(gate.m1.GROUP_ORDER)]
    return gate.m1.Cell(
        identifier=str(100000 + index),
        group=group,
        layer="MOp5",
        hemisphere="left",
        soma_um=(1.0, 2.0, 3.0),
        soma_distance_mm=np.linspace(0.1, 1.6, 16),
        k=8,
        observed_index=17,
        segment_unmapped=0,
    )


class ActualKProjectionTests(unittest.TestCase):
    def _write(self, directory: Path, name: str, rows: list[list[str]]) -> Path:
        path = directory / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\r\n")
            writer.writerows(rows)
        return path

    def test_cli_exposes_no_realized_membership_m0_swc_or_role_route(self) -> None:
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
                "--interface-root",
                "--geometry-root",
                "--target-free-root",
                "--target-root",
                "--output-root",
            },
        )
        for forbidden in ("m0", "swc", "membership", "final", "unassigned"):
            self.assertFalse(any(forbidden in option.lower() for option in options))

    def test_only_id_and_k_affect_projection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            left = self._write(
                root,
                "left.csv",
                [
                    TARGET_HEADER,
                    _target_row("100001", "3", 'secret,comma\nnewline "quote"'),
                    _target_row("100002", "16", "another sealed value"),
                ],
            )
            right = self._write(
                root,
                "right.csv",
                [
                    TARGET_HEADER,
                    _target_row("100001", "3", "completely different"),
                    _target_row("100002", "16", 'different,\n"quoted" secret'),
                ],
            )
            expected = {"100001", "100002"}
            self.assertEqual(
                gate._read_actual_k_projection(left, expected),
                gate._read_actual_k_projection(right, expected),
            )
            source = inspect.getsource(gate._scan_actual_k_projection)
            self.assertNotIn("csv.reader", source)
            self.assertNotIn("DictReader", source)

    def test_row_order_is_irrelevant_and_duplicate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            ordered = self._write(
                root,
                "ordered.csv",
                [TARGET_HEADER, _target_row("100001", "0"), _target_row("100002", "16")],
            )
            shuffled = self._write(
                root,
                "shuffled.csv",
                [TARGET_HEADER, _target_row("100002", "16"), _target_row("100001", "0")],
            )
            duplicate = self._write(
                root,
                "duplicate.csv",
                [TARGET_HEADER, _target_row("100001", "0"), _target_row("100001", "1")],
            )
            expected = {"100001", "100002"}
            self.assertEqual(
                gate._read_actual_k_projection(ordered, expected),
                gate._read_actual_k_projection(shuffled, expected),
            )
            with self.assertRaises(gate.ActualKGateError):
                gate._read_actual_k_projection(duplicate, {"100001"})

    def test_missing_or_noncanonical_k_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            wrong_header = list(TARGET_HEADER)
            wrong_header[gate.TARGET_K_INDEX] = "other"
            cases = (
                [wrong_header, _target_row("100001", "2")],
                [TARGET_HEADER, _target_row("100001", "17")],
                [TARGET_HEADER, _target_row("100001", "2.0")],
                [TARGET_HEADER, _target_row("100001", " 2")],
                [TARGET_HEADER, _target_row("100001", "2 ")],
            )
            for index, rows in enumerate(cases):
                path = self._write(root, f"bad_{index}.csv", rows)
                with self.assertRaises(gate.ActualKGateError):
                    gate._read_actual_k_projection(path, {"100001"})

    def test_malformed_quote_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            valid = self._write(
                root,
                "valid.csv",
                [TARGET_HEADER, _target_row("100001", "2", "sealed")],
            )
            malformed = root / "malformed.csv"
            data = valid.read_bytes().replace(b"sealed", b'"unterminated', 1)
            malformed.write_bytes(data)
            with self.assertRaises(gate.ActualKGateError):
                gate._read_actual_k_projection(malformed, {"100001"})
            linked = root / "linked.csv"
            linked.symlink_to(valid)
            with self.assertRaises(OSError):
                gate._read_actual_k_projection(linked, {"100001"})

    def test_200_to_194_projection_requires_six_blank_k_extras(self) -> None:
        cells = tuple(_cell(index) for index in range(gate.EXPECTED_ROWS))
        rows = [TARGET_HEADER]
        expected_k: dict[str, int] = {}
        for index, cell in enumerate(cells):
            value = index % 17
            expected_k[cell.identifier] = value
            rows.append(_target_row(cell.identifier, str(value), "sealed"))
        rows.extend(_target_row(f"excluded_{index}", "", "sealed") for index in range(6))
        with tempfile.TemporaryDirectory() as raw:
            path = self._write(Path(raw), "projection.csv", rows)
            projected, counts = gate._read_actual_k_projection(path, set(expected_k))
            self.assertEqual(projected, expected_k)
            self.assertEqual(counts.total_rows, 200)
            self.assertEqual(counts.matched_rows, 194)
            self.assertEqual(counts.extra_rows, 6)
            rows[-1][gate.TARGET_K_INDEX] = "1"
            bad = self._write(Path(raw), "bad_extra.csv", rows)
            with self.assertRaises(gate.ActualKGateError):
                gate._read_actual_k_projection(bad, set(expected_k))

    def test_exact_identity_join_and_inert_observed_index(self) -> None:
        cells = tuple(_cell(index) for index in range(gate.EXPECTED_ROWS))
        projected = {
            cell.identifier: index % 17 for index, cell in enumerate(reversed(cells))
        }
        aligned = gate._align_actual_k(cells, projected)
        self.assertEqual([cell.identifier for cell in aligned], [cell.identifier for cell in cells])
        self.assertEqual([cell.k for cell in aligned], [projected[cell.identifier] for cell in cells])
        library = gate.m1.CandidateLibrary()
        for before, after in zip(cells, aligned, strict=True):
            self.assertEqual(before.identifier, after.identifier)
            self.assertEqual(before.group, after.group)
            self.assertEqual(before.soma_um, after.soma_um)
            self.assertEqual(after.observed_index, library.basis(after.k).count)
            self.assertFalse(0 <= after.observed_index < library.basis(after.k).count)
        with self.assertRaises(gate.ActualKGateError):
            gate._align_actual_k(cells, dict(list(projected.items())[:-1]))
        extra = dict(projected)
        extra["999999"] = 1
        with self.assertRaises(gate.ActualKGateError):
            gate._align_actual_k(cells, extra)


class ActualKReportTests(unittest.TestCase):
    def test_aggregate_report_contains_no_fold_rows_ids_or_k_values(self) -> None:
        fold = {
            "fold": "must_not_survive",
            "training_cells": 150,
            "joint": {"rank": 261, "condition_number_2": 1234.0, "finite": True},
            "geometry_block": {"rank": 6, "finite": True},
            "geometry_residual": {"rank": 6, "finite": True},
            "status": "pass",
        }
        result = gate._aggregate_rank_result(
            {"status": "pass", "folds": [dict(fold) for _ in range(79)]}
        )
        self.assertEqual(result["training_folds_evaluated"], 79)
        self.assertEqual(result["training_folds_passed"], 79)
        self.assertFalse(result["fold_level_rows_emitted"])
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("must_not_survive", serialized)
        self.assertNotIn("provider_neuron_id", serialized)
        self.assertNotIn("realized_membership", serialized)

    def test_aggregate_pass_is_recomputed_from_every_frozen_threshold(self) -> None:
        fold = {
            "training_cells": 150,
            "joint": {"rank": 261, "condition_number_2": 1234.0, "finite": True},
            "geometry_block": {"rank": 6, "finite": True},
            "geometry_residual": {"rank": 6, "finite": True},
            "status": "pass",
        }
        base = {"status": "pass", "folds": [copy.deepcopy(fold) for _ in range(79)]}
        mutations = (
            ("joint", "rank", 260),
            ("geometry_block", "rank", 5),
            ("geometry_residual", "rank", 5),
            ("joint", "condition_number_2", 1.0e10 + 1.0),
            ("joint", "finite", False),
        )
        for section, key, value in mutations:
            candidate = copy.deepcopy(base)
            candidate["folds"][0][section][key] = value
            self.assertEqual(gate._aggregate_rank_result(candidate)["status"], "fail")
        candidate = copy.deepcopy(base)
        candidate["folds"][0]["status"] = "fail"
        self.assertEqual(gate._aggregate_rank_result(candidate)["status"], "fail")

    def test_frozen_fold_helper_still_has_exact_79_sets(self) -> None:
        cells = tuple(_cell(index) for index in range(gate.EXPECTED_ROWS))
        # Give every frozen group at least one row; only group membership enters
        # the fold iterator.
        folds = gate.m1._all_training_folds(cells)
        self.assertEqual(len(folds), gate.EXPECTED_FOLDS)
        self.assertEqual(len({indexes for _, indexes in folds}), gate.EXPECTED_FOLDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
