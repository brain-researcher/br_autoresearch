#!/usr/bin/env python3
"""Synthetic access-order and score tests for the EP10 M1/M0 comparator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


RUNNER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "compare_v3_m1_m0_v1.py"
)
LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "compare_v3_m1_m0_v1.sbatch"
)
SPEC = importlib.util.spec_from_file_location(
    "ep10_compare_v3_m1_m0_v1_tested", RUNNER_PATH
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import comparator: {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _outer_folds() -> list[dict[str, object]]:
    return [
        {
            "held_out_group": group,
            "held_out_cells": runner.GROUP_SIZES[group],
            "training_groups": 11,
            "inner_folds": 11,
            "selected_lambda_is_grid_boundary": False,
        }
        for group in runner.GROUP_ORDER
    ]


def _prediction_rows(model: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    ordinal = 0
    for group_index, group in enumerate(runner.GROUP_ORDER):
        for _ in range(runner.GROUP_SIZES[group]):
            if ordinal == 0:
                k = 0
            elif ordinal == 1:
                k = runner.PANEL_SIZE
            else:
                k = 1
            uniform = -math.log2(math.comb(runner.PANEL_SIZE, k))
            if k in (0, runner.PANEL_SIZE):
                score = 0.0
            elif model == "M0":
                score = uniform + 0.02 * group_index
            else:
                score = uniform + 0.03 * group_index
            rows.append(
                {
                    "provider_neuron_id": f"n{ordinal:03d}",
                    "fMOST_brain_id": group,
                    "K": str(k),
                    f"{model}_outer_cross_fitted_log2_probability": format(
                        score, ".17g"
                    ),
                    "uniform_fixed_K_log2_probability": format(uniform, ".17g"),
                }
            )
            ordinal += 1
    if ordinal != runner.EXPECTED_ROWS:
        raise AssertionError("synthetic frozen group vector changed")
    return rows


class ComparatorFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.scratch = self.base / "scratch"
        self.scratch.mkdir()
        self.scratch_episode = (
            self.scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
        )
        self.scratch_episode.mkdir(parents=True)
        self.stage0 = self.base / "stage0"
        self.stage0.mkdir()
        self.contract_path = self.stage0 / "v3_m1_model_contract.yaml"
        self.readiness_path = self.stage0 / "readiness_status.yaml"
        self.m1_root = self.scratch_episode / runner.M1_ROOT_BASENAME
        self.m0_root = self.scratch_episode / runner.M0_ROOT_BASENAME
        self.m1_root.mkdir()
        self.m0_root.mkdir()
        self.job_id = "999001"
        self.output_root = (
            self.scratch_episode / f"{runner.OUTPUT_PREFIX}_job_{self.job_id}"
        )
        self.args = SimpleNamespace(
            execute=True,
            contract=self.contract_path,
            readiness=self.readiness_path,
            m1_root=self.m1_root,
            m0_root=self.m0_root,
            output_root=self.output_root,
        )
        self.contract = {
            "schema_version": "episode10_v3_m1_model_v1",
            "status": "frozen_before_implementation_and_real_M1_execution",
            "sealed_scope": {
                "role": "development_only",
                "required_cells": 194,
                "required_groups": 12,
                "final_or_unassigned_access": "forbidden",
            },
            "required_inputs": {
                "M0_comparator_root": {
                    "required_basename": runner.M0_ROOT_BASENAME,
                    "required_file": runner.M0_PREDICTIONS_FILENAME,
                    "required_score_column": (
                        "M0_outer_cross_fitted_log2_probability"
                    ),
                    "required_join_columns": list(runner.PAIRING_KEYS),
                    "required_role": (
                        "outer_cross_fitted_development_predictions_only"
                    ),
                },
                "join": {
                    "key": "provider_neuron_id",
                    "cardinality": "one_to_one",
                    "exact_identity_set_required": True,
                    "exact_group_match_required": True,
                },
                "expected_group_sizes": dict(runner.GROUP_SIZES),
            },
        }
        self.readiness = {
            "launch": {
                "slurm_job_ids": [int(runner.M0_JOB_ID), int(runner.M1_JOB_ID)]
            },
            "access": {
                "v3_m1_M0_artifact_opened": runner.HISTORICAL_M0_ACCESS,
                "final_group_outcomes_opened": False,
            },
            "gates": {
                "gao_development_v3_m0_fit_v7": runner.M0_GATE,
                runner.POSTFLIGHT_KEY: (
                    runner.POSTFLIGHT_GATE
                ),
            },
        }
        self.m1_manifest = {
            "status": "complete",
            "stage": runner.M1_STAGE,
            "phase": "complete",
            "SLURM_job_id": runner.M1_JOB_ID,
            "development_cells": 194,
            "development_groups": 12,
            "completed_warm_paths": 78,
            "completed_zero_refits": 13,
            "M0_prediction_or_fit_artifact_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
        }
        self.m1_summary = {
            "status": "pass",
            "stage": runner.M1_STAGE,
            "analysis_label": "exploratory",
            "estimand": runner.ESTIMAND,
            "development_cells": 194,
            "development_groups": 12,
            "target_panel_size": 16,
            "outer_boundary_fold_count": 0,
            "outer_folds": _outer_folds(),
            "M0_prediction_or_fit_artifact_opened": False,
            "SWC_opened": False,
            "final_or_unassigned_opened": False,
            "full_refit_used_as_performance_estimate": False,
        }
        self.m0_summary = {
            "status": "pass",
            "stage": runner.M0_STAGE,
            "analysis_label": "exploratory",
            "estimand": runner.ESTIMAND,
            "development_cells": 194,
            "development_groups": 12,
            "target_panel_size": 16,
            "outer_boundary_fold_count": 0,
            "outer_folds": _outer_folds(),
            "zeros_interpreted_as_absence_or_negative": False,
            "final_or_unassigned_opened": False,
            "temperature_diagnostic_applied_to_predictions_or_scores": False,
            "output_root": str(self.m0_root),
        }
        self.m1_rows = _prediction_rows("M1")
        self.m0_rows = _prediction_rows("M0")
        self._write_fixture()

        self.environment = mock.patch.dict(
            os.environ,
            {"SCRATCH": str(self.scratch), "SLURM_JOB_ID": self.job_id},
        )
        self.contract_constant = mock.patch.object(
            runner, "CONTRACT_PATH", self.contract_path
        )
        self.readiness_constant = mock.patch.object(
            runner, "READINESS_PATH", self.readiness_path
        )
        self.environment.start()
        self.contract_constant.start()
        self.readiness_constant.start()

    def tearDown(self) -> None:
        self.readiness_constant.stop()
        self.contract_constant.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def _write_yaml(self, path: Path, payload: object) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _write_json(self, path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _write_fixture(self) -> None:
        self._write_yaml(self.contract_path, self.contract)
        self._write_yaml(self.readiness_path, self.readiness)
        self._write_json(
            self.m1_root / runner.M1_MANIFEST_FILENAME,
            self.m1_manifest,
        )
        self._write_json(
            self.m1_root / runner.M1_SUMMARY_FILENAME,
            self.m1_summary,
        )
        self._write_csv(
            self.m1_root / runner.M1_PREDICTIONS_FILENAME,
            self.m1_rows,
        )
        self._write_json(
            self.m0_root / runner.M0_SUMMARY_FILENAME,
            self.m0_summary,
        )
        self._write_csv(
            self.m0_root / runner.M0_PREDICTIONS_FILENAME,
            self.m0_rows,
        )

    def _report(self) -> dict[str, object]:
        return json.loads(
            (self.output_root / runner.REPORT_FILENAME).read_text(encoding="utf-8")
        )

    def test_exact_194_row_equal_group_happy_path(self) -> None:
        report = runner.run_comparison(self.args)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["phase"], "complete")
        self.assertTrue(report["M0_prediction_or_fit_artifact_opened"])
        self.assertEqual(report["historical_M0_artifact_access"], runner.HISTORICAL_M0_ACCESS)
        self.assertFalse(report["M1_fitting_process_M0_artifact_opened"])
        expected_gain = math.fsum(0.01 * value for value in range(12)) / 12
        self.assertAlmostEqual(
            report[
                "aggregate_equal_group_paired_M1_minus_M0_gain_bits_per_cell"
            ],
            expected_gain,
            places=14,
        )
        global_gain = math.fsum(
            float(m1["M1_outer_cross_fitted_log2_probability"])
            - float(m0["M0_outer_cross_fitted_log2_probability"])
            for m1, m0 in zip(self.m1_rows, self.m0_rows)
        ) / runner.EXPECTED_ROWS
        self.assertNotAlmostEqual(expected_gain, global_gain, places=6)
        for index, group in enumerate(runner.GROUP_ORDER):
            self.assertAlmostEqual(
                report["per_group"][group][
                    "paired_M1_minus_M0_gain_bits_per_cell"
                ],
                0.01 * index,
                places=14,
            )
        self.assertEqual(
            report["degenerate_K_counts"],
            {
                "K_0": 1,
                "K_16": 1,
                "both_models_exact_zero_for_all_degenerate_rows": True,
            },
        )
        self.assertFalse(list(self.output_root.glob("*.partial.*")))

    def test_all_pre_m0_failures_make_zero_m0_calls(self) -> None:
        originals = (
            deepcopy(self.readiness),
            deepcopy(self.m1_manifest),
            deepcopy(self.m1_summary),
            deepcopy(self.m1_rows),
        )
        cases: list[tuple[str, object]] = []
        bad_readiness = deepcopy(self.readiness)
        bad_readiness["gates"][
            runner.POSTFLIGHT_KEY
        ] = "pending"
        cases.append(("readiness", bad_readiness))
        erased_history = deepcopy(self.readiness)
        erased_history["access"]["v3_m1_M0_artifact_opened"] = False
        cases.append(("readiness", erased_history))
        obsolete_postflight = deepcopy(self.readiness)
        obsolete_postflight["gates"][runner.POSTFLIGHT_KEY] = runner.POSTFLIGHT_GATE.replace("46332190", "45739355")
        cases.append(("readiness", obsolete_postflight))
        bad_manifest = deepcopy(self.m1_manifest)
        bad_manifest["phase"] = "outer_predictions_complete"
        cases.append(("manifest", bad_manifest))
        bad_summary = deepcopy(self.m1_summary)
        bad_summary["outer_boundary_fold_count"] = 1
        cases.append(("summary", bad_summary))
        bad_rows = deepcopy(self.m1_rows)
        bad_rows[2]["M1_outer_cross_fitted_log2_probability"] = "nan"
        cases.append(("predictions", bad_rows))

        for label, payload in cases:
            with self.subTest(label=label):
                self.readiness, self.m1_manifest, self.m1_summary, self.m1_rows = (
                    deepcopy(value) for value in originals
                )
                if label == "readiness":
                    self.readiness = payload
                elif label == "manifest":
                    self.m1_manifest = payload
                elif label == "summary":
                    self.m1_summary = payload
                else:
                    self.m1_rows = payload
                self._write_fixture()
                with mock.patch.object(
                    runner,
                    "_load_m0_artifacts",
                    side_effect=AssertionError("M0 must remain untouched"),
                ) as m0_loader:
                    with self.assertRaises(runner.ComparisonError):
                        runner.run_comparison(self.args)
                    m0_loader.assert_not_called()
                self.assertFalse(self.output_root.exists())

    def test_pairing_and_row_invariants_fail_closed(self) -> None:
        fields_m1 = list(self.m1_rows[0])
        row_cases: list[tuple[str, list[dict[str, str]]]] = []
        duplicate = deepcopy(self.m1_rows)
        duplicate[3]["provider_neuron_id"] = duplicate[2]["provider_neuron_id"]
        row_cases.append(("duplicate", duplicate))
        missing = deepcopy(self.m1_rows[:-1])
        row_cases.append(("missing", missing))
        group_size = deepcopy(self.m1_rows)
        group_size[2]["fMOST_brain_id"] = runner.GROUP_ORDER[1]
        row_cases.append(("group_size", group_size))
        invalid_k = deepcopy(self.m1_rows)
        invalid_k[2]["K"] = "17"
        row_cases.append(("invalid_k", invalid_k))
        nonfinite = deepcopy(self.m1_rows)
        nonfinite[2]["M1_outer_cross_fitted_log2_probability"] = "inf"
        row_cases.append(("nonfinite", nonfinite))
        positive = deepcopy(self.m1_rows)
        positive[2]["M1_outer_cross_fitted_log2_probability"] = "0.1"
        row_cases.append(("positive_log_probability", positive))
        uniform = deepcopy(self.m1_rows)
        uniform[2]["uniform_fixed_K_log2_probability"] = "-3.9"
        row_cases.append(("uniform", uniform))
        singleton = deepcopy(self.m1_rows)
        singleton[0]["M1_outer_cross_fitted_log2_probability"] = "1e-30"
        row_cases.append(("singleton", singleton))
        for label, rows in row_cases:
            with self.subTest(label=label):
                with self.assertRaises(runner.ComparisonError):
                    runner._parse_prediction_rows(rows, fields_m1, model="M1")

        valid_m1 = runner._parse_prediction_rows(
            self.m1_rows,
            fields_m1,
            model="M1",
        )
        valid_m0 = list(
            runner._parse_prediction_rows(
                self.m0_rows,
                list(self.m0_rows[0]),
                model="M0",
            )
        )
        comparison_cases = []
        comparison_cases.append(("missing_pair", valid_m0[:-1]))
        group_mismatch = deepcopy(valid_m0)
        group_mismatch[2] = runner.ScoreRow(
            group_mismatch[2].identifier,
            runner.GROUP_ORDER[1],
            group_mismatch[2].k,
            group_mismatch[2].log2_probability,
            group_mismatch[2].uniform_log2_probability,
        )
        comparison_cases.append(("group_mismatch", group_mismatch))
        k_mismatch = deepcopy(valid_m0)
        k_mismatch[2] = runner.ScoreRow(
            k_mismatch[2].identifier,
            k_mismatch[2].group,
            2,
            k_mismatch[2].log2_probability,
            -math.log2(math.comb(16, 2)),
        )
        comparison_cases.append(("k_mismatch", k_mismatch))
        baseline_mismatch = deepcopy(valid_m0)
        baseline_mismatch[2] = runner.ScoreRow(
            baseline_mismatch[2].identifier,
            baseline_mismatch[2].group,
            baseline_mismatch[2].k,
            baseline_mismatch[2].log2_probability,
            baseline_mismatch[2].uniform_log2_probability + 1.0e-10,
        )
        comparison_cases.append(("baseline_mismatch", baseline_mismatch))
        for label, m0_rows in comparison_cases:
            with self.subTest(label=label):
                with self.assertRaises(runner.ComparisonError):
                    runner.compare_rows(valid_m1, m0_rows)

    def test_wrong_m1_identity_and_seals_fail_before_m0(self) -> None:
        original_manifest = deepcopy(self.m1_manifest)
        original_summary = deepcopy(self.m1_summary)
        cases = []
        wrong_job = deepcopy(original_manifest)
        wrong_job["SLURM_job_id"] = "other"
        cases.append(("manifest_job", wrong_job, original_summary))
        wrong_status = deepcopy(original_summary)
        wrong_status["status"] = "fail"
        cases.append(("summary_status", original_manifest, wrong_status))
        wrong_stage = deepcopy(original_summary)
        wrong_stage["stage"] = "other"
        cases.append(("summary_stage", original_manifest, wrong_stage))
        opened = deepcopy(original_summary)
        opened["final_or_unassigned_opened"] = True
        cases.append(("summary_seal", original_manifest, opened))
        for label, manifest, summary in cases:
            with self.subTest(label=label):
                self.m1_manifest = deepcopy(manifest)
                self.m1_summary = deepcopy(summary)
                self._write_fixture()
                with mock.patch.object(runner, "_load_m0_artifacts") as m0_loader:
                    with self.assertRaises(runner.ComparisonError):
                        runner.run_comparison(self.args)
                    m0_loader.assert_not_called()
                self.assertFalse(self.output_root.exists())
        self.m1_manifest = original_manifest
        self.m1_summary = original_summary

    def test_wrong_m1_root_fails_before_m0(self) -> None:
        self.args.m1_root = self.scratch_episode / "wrong_m1_root"
        with mock.patch.object(runner, "_load_m0_artifacts") as m0_loader:
            with self.assertRaises(runner.ComparisonError):
                runner.run_comparison(self.args)
            m0_loader.assert_not_called()
        self.assertFalse(self.output_root.exists())

    def test_pre_m0_failure_record_stays_false(self) -> None:
        original_atomic = runner._atomic_json
        calls = 0

        def fail_first_write(path: Path, payload: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise runner.ComparisonError("injected pre-M0 report failure")
            original_atomic(path, payload)

        with mock.patch.object(runner, "_atomic_json", side_effect=fail_first_write):
            with mock.patch.object(runner, "_load_m0_artifacts") as m0_loader:
                with self.assertRaises(runner.ComparisonError):
                    runner.run_comparison(self.args)
                m0_loader.assert_not_called()
        report = self._report()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["phase"], "prerequisites_passed_before_M0")
        self.assertFalse(report["M0_prediction_or_fit_artifact_opened"])

    def test_m0_failure_record_keeps_access_truth(self) -> None:
        self.m0_summary["status"] = "fail"
        self._write_fixture()
        with self.assertRaises(runner.ComparisonError):
            runner.run_comparison(self.args)
        report = self._report()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["phase"], "M0_access_started")
        self.assertTrue(report["M0_prediction_or_fit_artifact_opened"])
        self.assertEqual(report["failure_class"], "ComparisonError")
        self.assertFalse(report["SWC_opened"])
        self.assertFalse(report["final_or_unassigned_opened"])

    def test_wrong_m0_basename_fails_only_after_access_marker(self) -> None:
        self.args.m0_root = self.scratch_episode / "wrong_m0_root"
        with self.assertRaises(runner.ComparisonError):
            runner.run_comparison(self.args)
        report = self._report()
        self.assertEqual(report["phase"], "M0_access_started")
        self.assertTrue(report["M0_prediction_or_fit_artifact_opened"])

    def test_m0_summary_folds_identity_and_seals_are_required(self) -> None:
        cases: list[tuple[str, dict[str, object]]] = []
        failed = deepcopy(self.m0_summary)
        failed["status"] = "fail"
        cases.append(("status", failed))
        wrong_rows = deepcopy(self.m0_summary)
        wrong_rows["development_cells"] = 193
        cases.append(("rows", wrong_rows))
        wrong_fold = deepcopy(self.m0_summary)
        wrong_fold["outer_folds"][0]["held_out_group"] = runner.GROUP_ORDER[1]
        cases.append(("fold", wrong_fold))
        opened = deepcopy(self.m0_summary)
        opened["final_or_unassigned_opened"] = True
        cases.append(("seal", opened))
        for label, summary in cases:
            with self.subTest(label=label):
                with self.assertRaises(runner.ComparisonError):
                    runner._validate_m0_summary(summary, self.m0_root)

    def test_output_job_binding_and_foreign_reuse_fail_before_m0(self) -> None:
        wrong_args = argparse.Namespace(**vars(self.args))
        wrong_args.output_root = self.scratch_episode / "wrong_output"
        with mock.patch.object(runner, "_load_m0_artifacts") as m0_loader:
            with self.assertRaises(runner.ComparisonError):
                runner.run_comparison(wrong_args)
            m0_loader.assert_not_called()
        self.output_root.mkdir()
        with mock.patch.object(runner, "_load_m0_artifacts") as m0_loader:
            with self.assertRaises(runner.ComparisonError):
                runner.run_comparison(self.args)
            m0_loader.assert_not_called()


class StaticSurfaceTests(unittest.TestCase):
    def test_cli_and_launcher_have_only_the_authorized_surface(self) -> None:
        parser = runner.build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
            if option not in {"-h", "--help"}
        }
        self.assertEqual(
            options,
            {
                "--execute",
                "--contract",
                "--readiness",
                "--m1-root",
                "--m0-root",
                "--output-root",
            },
        )
        forbidden = {
            "--target-root",
            "--interface-root",
            "--geometry-root",
            "--swc-root",
            "--final-root",
            "--unassigned-root",
            "--full-refit",
            "--telemetry",
        }
        self.assertFalse(options & forbidden)

        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --job-name=ep10-v3-m1-m0-compare-v1", launcher)
        self.assertIn("#SBATCH --time=00:10:00", launcher)
        self.assertIn("#SBATCH --cpus-per-task=1", launcher)
        self.assertIn("#SBATCH --mem=2G", launcher)
        self.assertIn("#SBATCH --partition=russpold,normal", launcher)
        self.assertNotIn("#SBATCH --constraint", launcher)
        self.assertIn(runner.M1_ROOT_BASENAME, launcher)
        self.assertIn(runner.M0_ROOT_BASENAME, launcher)
        self.assertIn('exec >>"$runtime_log" 2>&1', launcher)
        self.assertNotIn("tee ", launcher)
        self.assertNotIn("> >(", launcher)
        self.assertNotIn("watch ", launcher)
        self.assertNotIn("while ", launcher)
        self.assertNotIn("for ", launcher)
        for line in launcher.splitlines():
            if line.lstrip().startswith("test "):
                self.assertNotIn("m0_root", line)
                self.assertNotIn(runner.M0_ROOT_BASENAME, line)
        for variable in (
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ):
            self.assertIn(variable, launcher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
