#!/usr/bin/env python3
"""Lightweight target-free tests for the v4 falsifier timing supplement."""

from __future__ import annotations

import inspect
import json
import math
import os
import pickle
import sys
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


TEST_PATH = Path(__file__).resolve()
TOOLS_DIR = TEST_PATH.parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import benchmark_v3_m2_falsifier_timing_v1 as bench  # noqa: E402


def _mocked_uncertified_probe(directory: str) -> dict:
    """Reproduce the actual exception constructor in a real spawned process."""
    error = bench.m1.UncertifiedLineSearchFailure("mocked optimizer failure", {
        "ridge_lambda": 0.1,
        "optimizer_iteration": 7,
        "current_gradient_infinity_norm": 0.0002,
        "certificate": {"certificate_pass": False, "relative_gap_predicate_pass": False},
    })
    error.production_fit_kind = "permutation_in_memory_warm"
    error.production_fit_key = f"leave_two__{bench.m1.GROUP_ORDER[0]}__{bench.m1.GROUP_ORDER[1]}"

    def fail(fixture, *, progress):
        progress("permutation_M1_started", 0)
        raise error

    with (
        mock.patch.object(bench, "build_synthetic_fixture", return_value=object()),
        mock.patch.object(bench, "benchmark_permutation_unit", side_effect=fail),
    ):
        return dict(bench._run_probe("permutation", directory))


def _mocked_successful_probe(directory: str) -> dict:
    with (
        mock.patch.object(bench, "build_synthetic_fixture", return_value=object()),
        mock.patch.object(bench, "build_physical_scaffold", return_value=SimpleNamespace(folds=tuple(range(12)))),
        mock.patch.object(bench, "_rademacher_dictionary", return_value=object()),
        mock.patch.object(bench, "_run_rademacher_core_fold", return_value={"stable_rank": 1, "precursor_fits": 0}),
    ):
        return dict(bench._run_probe("core_1", directory))


class MeasurementTests(unittest.TestCase):
    def test_componentwise_max_serial_and_repeat_are_conservative(self) -> None:
        first = bench.Measurement(2.0, 3.0, 4.0, 5.0)
        second = bench.Measurement(7.0, 1.0, 8.0, 2.0)
        maximum = bench.componentwise_max((first, second))
        self.assertEqual(
            maximum,
            bench.Measurement(7.0, 3.0, 8.0, 5.0),
        )
        repeated = bench.repeat_conservative(maximum, 12)
        self.assertEqual(repeated.wall_seconds, 84.0)
        self.assertEqual(repeated.cpu_seconds, 36.0)
        self.assertEqual(repeated.peak_rss_gib, 8.0)
        self.assertEqual(repeated.scratch_gib, 60.0)
        serial = bench.serial_sum((first, second))
        self.assertEqual(serial.wall_seconds, 9.0)
        self.assertEqual(serial.cpu_seconds, 4.0)
        self.assertEqual(serial.peak_rss_gib, 8.0)
        self.assertEqual(serial.scratch_gib, 7.0)

    def test_one_dictionary_object_is_shared_across_exactly_12_folds(self) -> None:
        created: list[object] = []
        seen: list[object] = []

        def factory(seed: int) -> object:
            self.assertEqual(seed, 2026092801)
            value = object()
            created.append(value)
            return value

        def runner(dictionary: object, fold: int) -> dict[str, int]:
            seen.append(dictionary)
            self.assertIn(fold, range(bench.EXPECTED_FOLDS))
            return {"stable_rank": 1, "precursor_fits": 0}

        result = bench.measure_shared_dictionary_trial(
            2026092801,
            tuple(range(bench.EXPECTED_FOLDS)),
            factory,
            runner,
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(len(seen), bench.EXPECTED_FOLDS)
        self.assertTrue(all(value is created[0] for value in seen))
        self.assertEqual(result.fold_count, bench.EXPECTED_FOLDS)
        self.assertEqual(result.stable_rank_count, bench.EXPECTED_FOLDS)
        self.assertEqual(result.precursor_fit_count, 0)
        self.assertGreater(result.measurement.wall_seconds, 0.0)


class HybridAndPermutationTests(unittest.TestCase):
    def test_hybrid_fold_fits_exactly_one_same_objective_precursor(self) -> None:
        configuration = bench.R_HYBRID_PROBE[1]
        theta = np.zeros(bench.m1.PARAMETER_DIMENSION, dtype=float)
        physical_fold = SimpleNamespace(fixed_m1=SimpleNamespace(theta=theta))
        fixture = SimpleNamespace(
            candidates=object(),
            geometry=object(),
        )
        fitted = SimpleNamespace(prepared=object())
        scales = SimpleNamespace(lambda_reference=2.0)
        maximum = SimpleNamespace(value=3.0)
        precursor = object()
        hybrid_result = SimpleNamespace(beta=np.zeros(bench.m2.PAIR_DIMENSION))

        with (
            mock.patch.object(
                bench,
                "_prepare_rademacher_fold",
                return_value=(fitted, object()),
            ),
            mock.patch.object(
                bench.optional,
                "fold_low_rank_scales_synthetic_qualification",
                return_value=scales,
            ),
            mock.patch.object(
                bench.optional,
                "fit_low_rank_synthetic_qualification",
                return_value=precursor,
            ) as precursor_fit,
            mock.patch.object(
                bench.m2,
                "fold_lambda_maximum",
                return_value=maximum,
            ),
            mock.patch.object(
                bench.optional,
                "fit_sparse_plus_low_rank_synthetic_qualification",
                return_value=hybrid_result,
            ) as hybrid_fit,
            mock.patch.object(bench, "_score_rademacher_validation"),
        ):
            witness = bench._run_rademacher_hybrid_fold(
                fixture,
                physical_fold,
                object(),
                configuration,
            )

        self.assertEqual(precursor_fit.call_count, 1)
        self.assertEqual(hybrid_fit.call_count, 1)
        self.assertIs(hybrid_fit.call_args.args[6], precursor)
        precursor_correction = precursor_fit.call_args.kwargs["linear_correction"]
        hybrid_correction = hybrid_fit.call_args.kwargs["linear_correction"]
        self.assertIs(precursor_correction, hybrid_correction)
        self.assertEqual(np.count_nonzero(precursor_correction), 0)
        self.assertEqual(witness, {"stable_rank": 1, "precursor_fits": 1})

    def test_permutation_accounting_accepts_only_interior_complete_execution(self) -> None:
        valid = SimpleNamespace(
            warm_path_count=66,
            zero_refit_count=12,
            full_development_refit_run=False,
            boundary_count=5,
        )
        self.assertEqual(bench.validate_permutation_accounting(valid), 5)
        for changed in (
            {"warm_path_count": 65},
            {"zero_refit_count": 11},
            {"full_development_refit_run": True},
            {"boundary_count": 6},
        ):
            malformed = SimpleNamespace(**{**vars(valid), **changed})
            with self.assertRaises(bench.FalsifierTimingError):
                bench.validate_permutation_accounting(malformed)


class AggregateReportTests(unittest.TestCase):
    def _measurements(self) -> dict[str, bench.Measurement]:
        return {
            unit: bench.Measurement(float(index + 1), 1.0, 0.5, 0.0)
            for index, unit in enumerate(bench.REQUIRED_UNITS)
        }

    def test_report_has_exact_four_units_and_only_aggregate_payloads(self) -> None:
        report = bench.build_report(
            self._measurements(),
            {
                "core_probe_count": 5,
                "hybrid_internal_precursor_fits": 12,
            },
            {
                "warm_path_fits": 66,
                "zero_start_outer_refits": 12,
                "boundary_selections": 3,
            },
            {name: "1" for name in bench.NUMERIC_THREAD_VARIABLES},
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            tuple(report["measured_units"]),
            bench.REQUIRED_UNITS,
        )
        encoded = json.dumps(report, allow_nan=False)
        self.assertNotIn("observed_index", encoded)
        self.assertNotIn("membership_matrix", encoded)
        self.assertNotIn("recipient_to_donor", encoded)
        bench.assert_aggregate_safe(report)

    def test_raw_or_nonfinite_payloads_are_rejected(self) -> None:
        for payload in (
            {"observed_index": 1},
            {"safe": np.zeros(2)},
            {"safe": math.nan},
        ):
            with self.assertRaises(bench.FalsifierTimingError):
                bench.assert_aggregate_safe(payload)

    def test_module_has_no_real_outcome_loader_dependency(self) -> None:
        source = inspect.getsource(bench)
        for forbidden in (
            "load_target_free_real_shape",
            "v3_m2_observed_session_v1",
            "gate_v3_m2_actual_k",
            "load_cells_from_swc",
        ):
            self.assertNotIn(forbidden, source)

    def test_runtime_and_launcher_pin_one_numeric_thread(self) -> None:
        environment = {
            "SLURM_JOB_ID": "123",
            "SLURM_CPUS_PER_TASK": "1",
            **{name: "1" for name in bench.NUMERIC_THREAD_VARIABLES},
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                dict(bench.require_slurm_one_thread()),
                {name: "1" for name in bench.NUMERIC_THREAD_VARIABLES},
            )
        environment["SLURM_CPUS_PER_TASK"] = "2"
        with mock.patch.dict(os.environ, environment, clear=True):
            self.assertEqual(
                dict(bench.require_slurm_one_thread()),
                {name: "1" for name in bench.NUMERIC_THREAD_VARIABLES},
            )
        launcher = (
            TOOLS_DIR / "benchmark_v3_m2_falsifier_timing_v1.sbatch"
        ).read_text(encoding="utf-8")
        self.assertTrue(any(
            f"#SBATCH --cpus-per-task={count}" in launcher for count in (1, 8)
        ))
        self.assertIn("#SBATCH --mem=8G", launcher)
        self.assertIn("#SBATCH --time=48:00:00", launcher)
        for name in bench.NUMERIC_THREAD_VARIABLES:
            self.assertIn(f"export {name}=1", launcher)


class CheckpointOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = mock.patch.dict(os.environ, {
            "SLURM_JOB_ID": "mocked_test_only",
            "SLURM_CPUS_PER_TASK": "8",
            **{name: "1" for name in bench.NUMERIC_THREAD_VARIABLES},
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def payload(self, key: str) -> dict:
        if key != "permutation":
            witness = {
                "fold_count": 12, "stable_rank_count": 12,
                "precursor_fit_count": 12 if key == "hybrid" else 0,
            }
        else:
            witness = {
                "mapping_count": 1,
                "direct_seed": bench.PERMUTATION_SEED,
                "mapping_shared_by_all_fits": True,
                "warm_path_fits": 66,
                "zero_start_outer_refits": 12,
                "full_development_M1_fit_run": False,
                "boundary_selections": 2,
                "boundary_gate_maximum_allowed": 5,
                "configured_M2_folds": 12,
                "configured_model_family": "group_lasso",
                "own_fresh_M1_baseline_used": True,
                "fold_aggregation": "componentwise_max_times_12",
            }
        ordinal = bench.PROBE_KEYS.index(key) + 1
        return dict(bench._completed_probe_payload(
            key, bench.Measurement(float(ordinal), 9.0 - ordinal, 0.5, 0.0), witness,
        ))

    def test_parallel_dispatch_uses_spawn_and_exact_eight_full_probes(self) -> None:
        submitted = []
        case = self

        class MockPool:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def submit(self, function, key, directory):
                case.assertIs(function, bench._run_probe)
                submitted.append(key)
                future = Future()
                future.set_result(case.payload(key))
                return future

        with mock.patch.object(bench, "ProcessPoolExecutor", return_value=MockPool()) as pool:
            report = bench.run_benchmark(workers=8, checkpoint_directory=self.root)
        self.assertEqual(submitted, list(bench.PROBE_KEYS))
        self.assertEqual(pool.call_args.kwargs["max_workers"], 8)
        self.assertEqual(pool.call_args.kwargs["mp_context"].get_start_method(), "spawn")
        self.assertEqual(report["measured_units"][bench.UNIT_RADEMACHER_CORE]["wall_seconds"], 5.0)
        self.assertEqual(report["measured_units"][bench.UNIT_RADEMACHER_CORE]["cpu_seconds"], 8.0)
        self.assertEqual(report["measured_units"][bench.UNIT_RADEMACHER_LOW_RANK]["wall_seconds"], 6.0)
        self.assertEqual(report["rademacher_witness"]["stable_rank_104_folds"], 84)

    def test_resume_reuses_complete_probe_and_restarts_partial_probe(self) -> None:
        source = self.root / "previous"
        source.mkdir()
        bench._write_json_atomic(source / "core_1.completed.json", self.payload("core_1"))
        bench._write_json_atomic(source / "low_rank.progress.json", {
            "status": "incomplete", "phase": "fold_started", "completed_fold_count": 11,
        })
        called = []

        def run(key, directory):
            called.append(key)
            return self.payload(key)

        target = self.root / "new"
        with mock.patch.object(bench, "_run_probe", side_effect=run):
            report = bench.run_benchmark(
                checkpoint_directory=target, reuse_completed_from=source,
            )
        self.assertEqual(called, list(bench.PROBE_KEYS[1:]))
        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            json.loads((target / "core_1.completed.json").read_text()),
            self.payload("core_1"),
        )

    def test_incompatible_or_incomplete_completed_artifact_fails_closed(self) -> None:
        changes = (
            ("identity", "fixture_seed", -1),
            ("identity", "direct_seed", -1),
            ("witness", "fold_count", 11),
            ("witness", "stable_rank_count", 11),
            ("measurement", "peak_rss_semantics", "job_total"),
        )
        for container, field, value in changes:
            with self.subTest(field=field):
                payload = self.payload("core_1")
                payload[container][field] = value
                bench._write_json_atomic(self.root / "core_1.completed.json", payload)
                with mock.patch.object(bench, "_run_probe") as run:
                    with self.assertRaises(bench.FalsifierTimingError):
                        bench.run_benchmark(
                            checkpoint_directory=self.root,
                            reuse_completed_from=self.root,
                        )
                    run.assert_not_called()

    def test_complete_reuse_needs_no_worker_or_fixture(self) -> None:
        for key in bench.PROBE_KEYS:
            bench._write_json_atomic(self.root / f"{key}.completed.json", self.payload(key))
        with (
            mock.patch.object(bench, "ProcessPoolExecutor") as pool,
            mock.patch.object(bench, "build_synthetic_fixture") as fixture,
        ):
            report = bench.run_benchmark(
                workers=8, checkpoint_directory=self.root, reuse_completed_from=self.root,
            )
        pool.assert_not_called()
        fixture.assert_not_called()
        self.assertEqual(report["status"], "pass")

    def test_missing_probe_cannot_produce_pass_report(self) -> None:
        payloads = {key: self.payload(key) for key in bench.PROBE_KEYS[:-1]}
        with self.assertRaises(bench.FalsifierTimingError):
            bench._assemble_probe_reports(payloads, {name: "1" for name in bench.NUMERIC_THREAD_VARIABLES})

    def test_worker_saves_only_complete_aggregate_result(self) -> None:
        fixture = object()
        cache = SimpleNamespace(folds=tuple(range(12)))
        dictionary = object()
        with (
            mock.patch.object(bench, "build_synthetic_fixture", return_value=fixture),
            mock.patch.object(bench, "build_physical_scaffold", return_value=cache),
            mock.patch.object(bench, "_rademacher_dictionary", return_value=dictionary) as factory,
            mock.patch.object(bench, "_run_rademacher_low_rank_fold", return_value={
                "stable_rank": 1, "precursor_fits": 0,
            }) as runner,
        ):
            payload = bench._run_probe("low_rank", str(self.root))
        factory.assert_called_once()
        self.assertEqual(runner.call_count, 12)
        self.assertTrue(all(call.args[2] is dictionary for call in runner.call_args_list))
        self.assertEqual(json.loads((self.root / "low_rank.completed.json").read_text()), payload)
        progress = json.loads((self.root / "low_rank.progress.json").read_text())
        self.assertEqual(progress["completed_fold_count"], 12)
        self.assertEqual(progress["phase"], "completed")
        bench.assert_aggregate_safe(payload)

    def test_failed_worker_retains_stage_without_completion(self) -> None:
        with mock.patch.object(bench, "build_synthetic_fixture", side_effect=MemoryError):
            failure = bench._run_probe("core_1", str(self.root))
        self.assertFalse((self.root / "core_1.completed.json").exists())
        progress = json.loads((self.root / "core_1.progress.json").read_text())
        self.assertEqual(progress["failed_during"], "fixture_started")
        self.assertEqual(progress["exception_type"], "MemoryError")
        self.assertEqual(progress, failure)

    def test_actual_solver_exception_requires_safe_transport_envelope(self) -> None:
        error = bench.m1.UncertifiedLineSearchFailure("mocked failure", {"optimizer_iteration": 7})
        with self.assertRaises(TypeError):
            pickle.loads(pickle.dumps(error))
        failure = _mocked_uncertified_probe(str(self.root))
        self.assertEqual(pickle.loads(pickle.dumps(failure)), failure)
        self.assertEqual(failure["exception_type"], "UncertifiedLineSearchFailure")
        self.assertEqual(failure["production_fit_ordinal"], 1)
        self.assertEqual(failure["optimizer_scalar_context"]["optimizer_iteration"], 7)
        self.assertEqual(failure["failed_during"], "permutation_M1_started")
        self.assertFalse((self.root / "permutation.completed.json").exists())
        self.assertEqual(json.loads((self.root / "permutation.progress.json").read_text()), failure)

    def test_spawned_numerical_failure_leaves_pool_usable(self) -> None:
        with bench.ProcessPoolExecutor(
            max_workers=2, mp_context=bench.multiprocessing.get_context("spawn"),
        ) as pool:
            failure_future = pool.submit(_mocked_uncertified_probe, str(self.root))
            failure = failure_future.result(timeout=30)
            # Submit after receiving the genuine solver failure: a broken pool
            # would reject this independent probe rather than preserve it.
            success = pool.submit(_mocked_successful_probe, str(self.root)).result(timeout=30)
        self.assertEqual(failure["status"], "failed")
        bench._validate_completed_probe("core_1", success)
        self.assertTrue((self.root / "core_1.completed.json").exists())

    def test_parent_collects_other_probes_then_emits_blocked_summary(self) -> None:
        failure = _mocked_uncertified_probe(str(self.root))
        collected = []
        # Inject a failure at the first collection slot to exercise continued
        # collection, independently of numerical probe scheduling order.
        early_failure = {**failure, "probe": "core_1"}

        def early_run(key, directory):
            collected.append(key)
            return early_failure if key == "core_1" else self.payload(key)

        with mock.patch.object(bench, "_run_probe", side_effect=early_run):
            with self.assertRaises(bench.FalsifierTimingError) as caught:
                bench.run_benchmark(checkpoint_directory=self.root / "aggregate")
        self.assertEqual(collected, list(bench.PROBE_KEYS))
        report = bench._blocked_report(caught.exception)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["completed_probe_count"], 7)
        self.assertEqual(report["probe_failures"], [early_failure])

    def test_solver_failure_context_rejects_vectors_and_prohibited_fields(self) -> None:
        for context in ({"vector": np.zeros(2)}, {"observed_index": 1}):
            error = bench.m1.UncertifiedLineSearchFailure("mocked failure", context)
            failure = bench._probe_failure_payload("permutation", "permutation_M1_started", 0, error)
            self.assertEqual(failure["optimizer_scalar_context_status"], "rejected_nonaggregate")
            self.assertNotIn("optimizer_scalar_context", failure)
            bench.assert_aggregate_safe(failure)

    def test_progress_is_outside_measurement_and_fold_projection_unchanged(self) -> None:
        measuring = False
        calls = 0
        events = []

        def time_call(function):
            nonlocal measuring, calls
            calls += 1
            measuring = True
            try:
                result = function()
                return bench.TimedResult(result, bench.Measurement(float(calls), 1.0, 0.5))
            finally:
                measuring = False

        def progress(phase, completed):
            self.assertFalse(measuring)
            events.append((phase, completed))

        with mock.patch.object(bench, "_time_call", side_effect=time_call):
            result = bench.measure_shared_dictionary_trial(
                2026092801, tuple(range(12)), lambda seed: object(),
                lambda dictionary, fold: {"stable_rank": 1, "precursor_fits": 0},
                progress=progress,
            )
        self.assertEqual(result.measurement.wall_seconds, 1.0 + 13.0 * 12)
        self.assertEqual(result.measurement.cpu_seconds, 1.0 + 1.0 * 12)
        self.assertEqual(events[-1], ("fold_completed", 12))

    def test_workers_cannot_exceed_allocation_or_probe_count(self) -> None:
        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "2"}):
            with self.assertRaises(bench.FalsifierTimingError):
                bench.run_benchmark(workers=8, checkpoint_directory=self.root)
        with self.assertRaises(bench.FalsifierTimingError):
            bench.run_benchmark(workers=9, checkpoint_directory=self.root)


if __name__ == "__main__":
    unittest.main()
