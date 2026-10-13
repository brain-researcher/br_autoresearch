#!/usr/bin/env python3
"""Focused synthetic orchestration tests for the EP10 v3 real-M1 runner."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "fit_v3_m1_real_v1.py"
LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "fit_v3_m1_real_single_thread_v3.sbatch"
)
SPEC = importlib.util.spec_from_file_location("ep10_fit_v3_m1_real_v1_tested", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import real-M1 runner: {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _execution_identity() -> dict[str, Any]:
    return {
        "numerical_execution_schema_version": runner.EXECUTION_SCHEMA_VERSION,
        "qualification_job_id": runner.QUALIFICATION_JOB_ID,
        "prior_membership_opening_job_id": runner.PRIOR_MEMBERSHIP_OPENING_JOB_ID,
        "cpu_model": "AMD EPYC 7502 32-Core Processor",
        "numpy": "1.26.4",
        "openblas_library": "libopenblas-test.so",
        "openblas_config": "OpenBLAS test Zen",
        "openblas_core": "Zen",
        "openblas_threads": 1,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
    }


def _target_header() -> list[str]:
    fields = [f"sealed_field_{index}" for index in range(runner.TARGET_FIELD_COUNT)]
    fields[runner.TARGET_IDENTIFIER_INDEX] = "provider_neuron_id"
    fields[runner.TARGET_GROUP_INDEX] = "fMOST_brain_id"
    fields[runner.TARGET_K_INDEX] = "K"
    for index, target in enumerate(runner.m1.PANEL):
        fields[runner.TARGET_DETECTED_START_INDEX + index] = f"detected__{target}"
    return fields


def _target_row(
    identifier: str,
    k: str,
    detected: Sequence[str],
    *,
    group: str = "212074",
    sealed_value: str = "",
) -> list[str]:
    row = [""] * runner.TARGET_FIELD_COUNT
    row[runner.TARGET_IDENTIFIER_INDEX] = identifier
    row[runner.TARGET_GROUP_INDEX] = group
    row[runner.TARGET_K_INDEX] = k
    row[9] = sealed_value
    for index, value in enumerate(detected):
        row[runner.TARGET_DETECTED_START_INDEX + index] = value
    return row


def _csv_bytes(rows: Sequence[Sequence[str]]) -> io.BytesIO:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\r\n")
    writer.writerows(rows)
    return io.BytesIO(text.getvalue().encode("utf-8"))


def _fake_cells() -> tuple[Any, ...]:
    cells: list[Any] = []
    ordinal = 0
    for group in runner.m1.GROUP_ORDER:
        for _ in range(runner.m1.GROUP_SIZES[group]):
            cells.append(
                SimpleNamespace(
                    identifier=str(100000 + ordinal),
                    group=group,
                    k=1,
                    observed_index=0,
                )
            )
            ordinal += 1
    if len(cells) != runner.EXPECTED_ROWS:
        raise AssertionError("synthetic orchestration fixture changed")
    return tuple(cells)


def _optimizer(ridge_lambda: float) -> Any:
    theta = np.zeros(runner.m1.PARAMETER_DIMENSION, dtype=float)
    theta[0] = math.log10(ridge_lambda)
    return runner.m1.OptimizerResult(
        theta=theta,
        objective=1.0,
        gradient_infinity_norm=0.0,
        iterations=0,
        accepted_steps=0,
        relative_objective_change=0.0,
        telemetry={
            "status": "pass",
            "ridge_lambda": ridge_lambda,
            "convergence_witness": "gradient_and_repeated_unchanged_initial_objective",
        },
    )


class FakeFactories:
    def __init__(self, loss: Callable[[float], float]) -> None:
        self.loss = loss
        self.warm_calls: list[str] = []
        self.zero_calls: list[tuple[str, float]] = []

    def warm(
        self,
        key: str,
        indexes: Sequence[int],
        cells: Sequence[Any],
        moments: Sequence[Any],
        library: Any,
        geometry: Any,
        checkpoint_root: Path,
    ) -> Any:
        del moments, library, geometry, checkpoint_root
        self.warm_calls.append(key)
        prepared = SimpleNamespace(cells=tuple(cells[index] for index in indexes))
        fits = {
            ridge_lambda: _optimizer(ridge_lambda)
            for ridge_lambda in runner.m1.LAMBDAS_DESCENDING
        }
        return runner.TrainingBundle(key, tuple(indexes), prepared, fits, False)

    def zero(
        self,
        key: str,
        indexes: Sequence[int],
        ridge_lambda: float,
        cells: Sequence[Any],
        moments: Sequence[Any],
        library: Any,
        geometry: Any,
        checkpoint_root: Path,
    ) -> Any:
        del moments, library, geometry, checkpoint_root
        self.zero_calls.append((key, ridge_lambda))
        prepared = SimpleNamespace(cells=tuple(cells[index] for index in indexes))
        return runner.TrainingBundle(
            key,
            tuple(indexes),
            prepared,
            {ridge_lambda: _optimizer(ridge_lambda)},
            False,
        )

    def predict(
        self,
        bundle: Any,
        result: Any,
        validation_indexes: Sequence[int],
        cells: Sequence[Any],
        moments: Sequence[Any],
        library: Any,
        geometry: Any,
    ) -> Any:
        del bundle, moments, library, geometry
        ridge_lambda = float(result.telemetry["ridge_lambda"])
        size = len(validation_indexes)
        log_probability = np.full(size, -self.loss(ridge_lambda), dtype=float)
        marginals = np.zeros((size, runner.m1.PANEL_SIZE), dtype=float)
        for row, index in enumerate(validation_indexes):
            marginals[row, :] = float(cells[index].k) / runner.m1.PANEL_SIZE
        return runner.Prediction(log_probability, marginals)


class MembershipProjectionTests(unittest.TestCase):
    def test_only_id_group_k_and_detected16_affect_projection(self) -> None:
        first = ["0"] * runner.m1.PANEL_SIZE
        first[0] = "1"
        first[5] = "1"
        second = ["0"] * runner.m1.PANEL_SIZE
        second[15] = "1"
        rows_left = [
            _target_header(),
            _target_row("100001", "2", first, sealed_value='secret,\n"quote"'),
            _target_row("100002", "1", second, sealed_value="sealed A"),
            _target_row("excluded", "", [""] * 16, sealed_value="sealed excluded"),
        ]
        rows_right = [
            _target_header(),
            _target_row("100001", "2", first, sealed_value="different"),
            _target_row("100002", "1", second, sealed_value='other,\n"value"'),
            _target_row("excluded", "", [""] * 16, sealed_value="changed excluded"),
        ]
        expected = {"100001", "100002"}
        left = runner._scan_membership_projection(_csv_bytes(rows_left), expected)
        right = runner._scan_membership_projection(_csv_bytes(rows_right), expected)
        self.assertEqual(left, right)
        self.assertEqual(left[1], runner.MembershipScanCounts(3, 2, 1))
        self.assertEqual(
            left[0]["100001"],
            ("212074", 2, tuple(int(value) for value in first)),
        )
        source = inspect.getsource(runner._scan_membership_projection)
        self.assertNotIn("csv.reader", source)
        self.assertNotIn("DictReader", source)

    def test_k_membership_mismatch_and_nonblank_excluded_fail_closed(self) -> None:
        detected = ["0"] * runner.m1.PANEL_SIZE
        detected[3] = "1"
        mismatch = [
            _target_header(),
            _target_row("100001", "2", detected),
        ]
        with self.assertRaises(runner.RealM1Error):
            runner._scan_membership_projection(_csv_bytes(mismatch), {"100001"})
        excluded = [
            _target_header(),
            _target_row("100001", "1", detected),
            _target_row("excluded", "", ["1", *("" for _ in range(15))]),
        ]
        with self.assertRaises(runner.RealM1Error):
            runner._scan_membership_projection(_csv_bytes(excluded), {"100001"})

    def test_header_order_and_binary_values_fail_closed(self) -> None:
        detected = ["0"] * runner.m1.PANEL_SIZE
        detected[0] = "1"
        wrong_header = _target_header()
        wrong_header[runner.TARGET_DETECTED_START_INDEX] = "other"
        cases = (
            [wrong_header, _target_row("100001", "1", detected)],
            [_target_header(), _target_row("100001", "1", ["2", *detected[1:]])],
            [_target_header(), _target_row("100001", "01", detected)],
        )
        for rows in cases:
            with self.assertRaises(runner.RealM1Error):
                runner._scan_membership_projection(_csv_bytes(rows), {"100001"})

    def test_alignment_replaces_k_and_observed_index_together(self) -> None:
        cells: list[Any] = []
        membership: dict[str, tuple[str, int, tuple[int, ...]]] = {}
        ordinal = 0
        for group in runner.m1.GROUP_ORDER:
            for _ in range(runner.m1.GROUP_SIZES[group]):
                identifier = str(100000 + ordinal)
                k = ordinal % 17
                detected = np.zeros(16, dtype=int)
                if k:
                    detected[:k] = 1
                cells.append(
                    runner.m1.Cell(
                        identifier=identifier,
                        group=group,
                        layer="MOp5",
                        hemisphere="left",
                        soma_um=(1.0, 2.0, 3.0),
                        soma_distance_mm=np.linspace(0.1, 1.6, 16),
                        k=8,
                        observed_index=1,
                        segment_unmapped=0,
                    )
                )
                membership[identifier] = (
                    group,
                    k,
                    tuple(int(value) for value in detected),
                )
                ordinal += 1
        library = runner.m1.CandidateLibrary()
        aligned = runner._align_membership(cells, membership, library)
        for cell in aligned:
            _, k, detected = membership[cell.identifier]
            self.assertEqual(cell.k, k)
            self.assertTrue(
                np.array_equal(
                    library.basis(k).membership[cell.observed_index],
                    np.asarray(detected, dtype=np.uint8),
                )
            )

        mismatched = dict(membership)
        first = cells[0]
        _, first_k, first_detected = mismatched[first.identifier]
        mismatched[first.identifier] = ("220187", first_k, first_detected)
        with self.assertRaisesRegex(runner.RealM1Error, "group mismatch"):
            runner._align_membership(cells, mismatched, library)


class OrchestrationTests(unittest.TestCase):
    def test_one_se_selects_largest_qualifying_lambda(self) -> None:
        validation: dict[float, list[float]] = {}
        for ridge_lambda in runner.m1.LAMBDAS_DESCENDING:
            if ridge_lambda == 1.0:
                values = [1.0, 1.2, 0.8]
            elif ridge_lambda == 10.0:
                values = [1.05, 1.25, 0.85]
            else:
                values = [2.0, 2.0, 2.0]
            validation[ridge_lambda] = values
        selected, details = runner.select_lambda_one_se(validation, 3)
        self.assertEqual(selected, 10.0)
        self.assertEqual(details["minimum_mean_lambda"], 1.0)

    def test_exact_nested_logo_reuse_and_zero_start_refits(self) -> None:
        cells = _fake_cells()
        moments = (None,) * len(cells)
        factories = FakeFactories(lambda value: abs(math.log10(value)))
        selection = runner.select_outer_lambdas(
            cells,
            moments,
            object(),
            object(),
            Path("/unused"),
            warm_factory=factories.warm,
            predictor=factories.predict,
        )
        self.assertEqual(len(factories.warm_calls), 66)
        self.assertEqual(len(set(factories.warm_calls)), 66)
        self.assertEqual(selection.boundary_count, 0)
        self.assertEqual(set(selection.selected_lambdas.values()), {1.0})
        self.assertTrue(all(detail["inner_folds"] == 11 for detail in selection.details))

        cross_fit = runner.refit_outer_and_predict(
            cells,
            moments,
            object(),
            object(),
            Path("/unused"),
            selection,
            zero_factory=factories.zero,
            predictor=factories.predict,
        )
        self.assertEqual(len(factories.zero_calls), 12)
        self.assertTrue(all(key.startswith("outer_refit__") for key, _ in factories.zero_calls))
        self.assertTrue(np.all(np.isfinite(cross_fit.prediction.log_probability_nats)))
        self.assertEqual(cross_fit.prediction.marginals.shape, (194, 16))

        full = runner.tune_and_refit_full_development(
            cells,
            moments,
            object(),
            object(),
            Path("/unused"),
            cross_fit.warm_cache,
            warm_factory=factories.warm,
            zero_factory=factories.zero,
            predictor=factories.predict,
        )
        self.assertEqual(len(factories.warm_calls), 78)
        self.assertEqual(len(set(factories.warm_calls)), 78)
        self.assertEqual(len(factories.zero_calls), 13)
        self.assertEqual(factories.zero_calls[-1], ("full_refit", 1.0))
        self.assertEqual(full.selected_lambda, 1.0)

    def test_boundary_stop_precedes_every_zero_start_refit(self) -> None:
        cells = _fake_cells()
        moments = (None,) * len(cells)
        factories = FakeFactories(lambda value: math.log10(1000.0 / value))
        selection = runner.select_outer_lambdas(
            cells,
            moments,
            object(),
            object(),
            Path("/unused"),
            warm_factory=factories.warm,
            predictor=factories.predict,
        )
        self.assertEqual(selection.boundary_count, 12)
        with self.assertRaises(runner.BoundaryReview):
            runner.refit_outer_and_predict(
                cells,
                moments,
                object(),
                object(),
                Path("/unused"),
                selection,
                zero_factory=factories.zero,
                predictor=factories.predict,
            )
        self.assertEqual(factories.zero_calls, [])


class CheckpointAndCliTests(unittest.TestCase):
    def test_output_root_same_job_bootstrap_and_completed_replay(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            scratch = Path(raw).resolve()
            episode = scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
            episode.mkdir(parents=True)
            job_id = "123456"
            output = episode / f"{runner.OUTPUT_PREFIX}_job_{job_id}"
            output.mkdir()
            (output / runner.CHECKPOINT_DIRECTORY).mkdir()
            previous_scratch = os.environ.get("SCRATCH")
            previous_job = os.environ.get("SLURM_JOB_ID")
            os.environ["SCRATCH"] = str(scratch)
            os.environ["SLURM_JOB_ID"] = job_id
            identity = _execution_identity()
            try:
                resolved, checkpoints, resumed, completed = runner._prepare_output_root(
                    output, identity
                )
                self.assertEqual(resolved, output)
                self.assertEqual(checkpoints, output / runner.CHECKPOINT_DIRECTORY)
                self.assertTrue(resumed)
                self.assertIsNone(completed)

                runner._write_checkpoint_manifest(
                    output,
                    execution_identity=identity,
                    phase="outer_predictions_complete_full_tuning",
                    realized_membership_opened=True,
                    completed_warm_paths=66,
                    completed_zero_refits=12,
                )
                summary = {
                    "status": "pass",
                    "stage": runner.PRODUCTION_STAGE,
                    "actual_K_gate_job_id": runner.ACTUAL_K_JOB_ID,
                    "execution_identity": identity,
                    "development_cells": runner.EXPECTED_ROWS,
                    "development_groups": runner.EXPECTED_GROUPS,
                    "M0_prediction_or_fit_artifact_opened": False,
                    "SWC_opened": False,
                    "final_or_unassigned_opened": False,
                }
                for name in runner.REQUIRED_OUTPUTS:
                    if name == runner.SUMMARY_FILENAME:
                        runner._atomic_json(output / name, summary)
                    else:
                        (output / name).write_text("synthetic\n", encoding="utf-8")
                _, _, resumed, completed = runner._prepare_output_root(
                    output, identity
                )
                self.assertTrue(resumed)
                self.assertEqual(completed, summary)
            finally:
                if previous_scratch is None:
                    os.environ.pop("SCRATCH", None)
                else:
                    os.environ["SCRATCH"] = previous_scratch
                if previous_job is None:
                    os.environ.pop("SLURM_JOB_ID", None)
                else:
                    os.environ["SLURM_JOB_ID"] = previous_job

    def test_checkpoint_round_trip_and_certificate_witness(self) -> None:
        ordinary = _optimizer(1.0)
        certificate = runner.m1.OptimizerResult(
            theta=np.zeros(runner.m1.PARAMETER_DIMENSION),
            objective=1.0,
            gradient_infinity_norm=1.0e-9,
            iterations=1,
            accepted_steps=1,
            relative_objective_change=math.inf,
            telemetry={
                "status": "pass",
                "ridge_lambda": 1.0e-3,
                "convergence_witness": "strong_convexity_gap_after_zoom_stall",
                "certificate": {
                    "certificate_pass": True,
                    "gradient_predicate_pass": True,
                    "relative_gap_predicate_pass": True,
                    "prior_relative_objective_change": None,
                },
            },
        )
        runner._validate_optimizer_result(ordinary, 1.0)
        runner._validate_optimizer_result(certificate, 1.0e-3)
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "warm__test.npz"
            previous_job = os.environ.get("SLURM_JOB_ID")
            os.environ["SLURM_JOB_ID"] = "123456"
            try:
                runner._write_fit_checkpoint(
                    path,
                    kind="warm",
                    key="test",
                    results={1.0: ordinary, 1.0e-3: certificate},
                    training_cells=10,
                    training_groups=("212074", "220187"),
                )
                restored = runner._load_fit_checkpoint(
                    path,
                    kind="warm",
                    key="test",
                    expected_lambdas=(1.0, 1.0e-3),
                    training_cells=10,
                    training_groups=("212074", "220187"),
                )
                self.assertEqual(tuple(restored), (1.0, 1.0e-3))
                self.assertTrue(
                    math.isinf(restored[1.0e-3].relative_objective_change)
                )
                os.environ["SLURM_JOB_ID"] = "654321"
                with self.assertRaises(runner.RealM1Error):
                    runner._load_fit_checkpoint(
                        path,
                        kind="warm",
                        key="test",
                        expected_lambdas=(1.0, 1.0e-3),
                        training_cells=10,
                        training_groups=("212074", "220187"),
                    )
            finally:
                if previous_job is None:
                    os.environ.pop("SLURM_JOB_ID", None)
                else:
                    os.environ["SLURM_JOB_ID"] = previous_job

    def test_cli_has_no_m0_swc_final_or_unassigned_route(self) -> None:
        options = {
            option
            for action in runner.build_parser()._actions
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
                "--actual-k-root",
                "--execution-addendum",
                "--qualification-root",
                "--affected-refresh-root",
                "--resource-amendment",
                "--resource-assessment",
                "--target-root",
                "--output-root",
            },
        )
        for forbidden in ("m0", "swc", "final", "unassigned"):
            self.assertFalse(any(forbidden in option.lower() for option in options))

    def test_passed_single_thread_qualification_and_backend_binding(self) -> None:
        scratch = Path(os.environ["SCRATCH"])
        root = (
            scratch
            / "br_autoresearch"
            / "episode10_single_cell_coprojection"
            / runner.QUALIFICATION_ROOT_BASENAME
        )
        summary = runner.require_single_thread_qualification(root)
        backend = dict(summary["numeric_backend"])
        runner.require_matching_qualification_backend(summary, backend)
        mismatched = dict(backend)
        mismatched["openblas_threads"] = 2
        with self.assertRaises(runner.RealM1Error):
            runner.require_matching_qualification_backend(summary, mismatched)

    def test_preaccess_order_and_single_thread_launcher(self) -> None:
        source = inspect.getsource(runner.run_real)
        ordered_calls = (
            "validate_real_contract(args.contract)",
            "execution_policy.validate_addendum(",
            "execution_policy.numeric_backend()",
            "require_single_thread_qualification(args.qualification_root)",
            "require_matching_qualification_backend(",
            "require_approved_numerical_refresh(",
            "require_actual_k_prerequisite(",
            "_prepare_output_root(",
            "_read_membership_inputs(",
        )
        positions = [source.index(value) for value in ordered_calls]
        self.assertEqual(positions, sorted(positions))
        self.assertIn(
            "membership_opened_by_job_id=PRIOR_MEMBERSHIP_OPENING_JOB_ID",
            source,
        )
        self.assertNotIn("if not failure_path.exists()", source)
        self.assertIn('"optimizer_scalar_context"', source)
        self.assertEqual(
            runner.CHECKPOINT_SCHEMA,
            "episode10_v3_m1_real_fit_checkpoint_single_thread_v3",
        )

        launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --cpus-per-task=1", launcher)
        self.assertIn("#SBATCH --time=12:00:00", launcher)
        self.assertIn("#SBATCH --constraint=CPU_SKU:7502", launcher)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertIn(f"export {name}=1", launcher)
        self.assertNotIn("$SLURM_CPUS_PER_TASK", launcher)
        self.assertNotIn("tee", launcher)
        self.assertNotIn(">(", launcher)
        self.assertIn(
            "v3_m1_real_fit_single_thread_v3_job_${SLURM_JOB_ID}", launcher
        )
        self.assertNotIn("v3_m1_real_fit_v1_job_${SLURM_JOB_ID}", launcher)
        self.assertIn("--execution-addendum", launcher)
        self.assertIn("--qualification-root", launcher)
        self.assertLess(
            launcher.index('"$python_bin" "$validator"'),
            launcher.index('"$python_bin" "$test_file"'),
        )

    def test_required_artifacts_and_work_accounting_are_frozen(self) -> None:
        self.assertEqual(
            runner.REQUIRED_OUTPUTS,
            (
                "v3_m1_cross_fitted_predictions.csv",
                "v3_m1_fit_summary.json",
                "v3_m1_full_refit.json",
                "v3_m1_solver_telemetry.json",
            ),
        )
        telemetry_source = inspect.getsource(runner._solver_telemetry)
        self.assertIn('"executed_conservative_path_equivalents_with_exact_reuse": 91', telemetry_source)
        self.assertIn('"frozen_uncached_budget_basis_path_equivalents": 157', telemetry_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
